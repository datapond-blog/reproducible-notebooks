"""Analysis views derived from the invoice-support simulator's OTLP JSON."""

import json
from pathlib import Path

import polars as pl


def decode_value(value: dict):
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [decode_value(item) for item in value["arrayValue"]["values"]]
    if "kvlistValue" in value:
        return decode_attributes(value["kvlistValue"]["values"])
    raise ValueError("Unsupported AnyValue in this example's loader")


def decode_attributes(values: list[dict]) -> dict:
    return {item["key"]: decode_value(item["value"]) for item in values}


def load_spans(path: str | Path) -> pl.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for resource in payload["resourceSpans"]:
        resource_attrs = decode_attributes(resource["resource"]["attributes"])
        for scope in resource["scopeSpans"]:
            for span in scope["spans"]:
                attrs = decode_attributes(span["attributes"])
                start_ns = int(span["startTimeUnixNano"])
                rows.append({
                    "trace_id": span["traceId"],
                    "span_id": span["spanId"],
                    "parent_span_id": span.get("parentSpanId"),
                    "service_name": resource_attrs["service.name"],
                    "span_name": span["name"],
                    "operation_name": attrs["gen_ai.operation.name"],
                    "start_ns": start_ns,
                    "duration_ms": (int(span["endTimeUnixNano"]) - start_ns) / 10**6,
                    "status_code": "ERROR" if (
                        span.get("status", {}).get("code") == 2 or attrs.get("error.type")
                    ) else "UNSET",
                    "input_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
                    "output_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
                    "attributes_json": json.dumps(attrs),
                })
    return pl.DataFrame(rows)


def summarize_invocations(spans: pl.DataFrame) -> pl.DataFrame:
    roots = spans.filter(pl.col("operation_name") == "invoke_agent")
    return roots.group_by("service_name").agg(
        pl.len().alias("invocations"),
        pl.col("duration_ms").quantile(0.95, interpolation="nearest").alias("p95_duration_ms"),
        (pl.col("status_code") == "ERROR").cast(pl.Float64).mean().alias("failure_rate"),
    ).sort("service_name")


def summarize_operations(spans: pl.DataFrame) -> pl.DataFrame:
    # Exclude the enclosing agent span: parent and child durations overlap.
    return spans.filter(pl.col("operation_name") != "invoke_agent").group_by(
        "service_name", "operation_name"
    ).agg(
        pl.len().alias("calls"),
        pl.col("duration_ms").median().alias("median_duration_ms"),
        (pl.col("status_code") == "ERROR").cast(pl.Float64).mean().alias("error_rate"),
    ).sort("service_name", "operation_name")


def invocation_details(spans: pl.DataFrame) -> pl.DataFrame:
    roots = spans.filter(pl.col("operation_name") == "invoke_agent").select(
        "trace_id", "service_name", "duration_ms", "status_code"
    )
    children = spans.filter(pl.col("operation_name") != "invoke_agent").group_by("trace_id").agg(
        (pl.col("operation_name") == "execute_tool").sum().alias("tool_attempts"),
        ((pl.col("operation_name") == "execute_tool") & (pl.col("status_code") == "ERROR"))
        .sum().alias("tool_errors"),
        (pl.col("input_tokens") + pl.col("output_tokens")).sum().alias("model_tokens"),
    )
    return roots.join(children, on="trace_id", how="left").with_columns(
        pl.when(pl.col("status_code") == "ERROR").then(pl.lit("failed"))
        .when(pl.col("tool_errors") > 0).then(pl.lit("recovered"))
        .otherwise(pl.lit("success")).alias("outcome")
    ).sort("service_name", "trace_id")
