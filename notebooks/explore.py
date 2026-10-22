import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    from reproducible_notebook.telemetry import (
        invocation_details, load_spans, summarize_invocations, summarize_operations,
    )

    return (
        invocation_details,
        load_spans,
        mo,
        pl,
        px,
        summarize_invocations,
        summarize_operations,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inside a simulated support agent

    One OTLP file contains three deployments of an invoice-support agent.
    Compare invocation latency, separate tool errors from agent failures, then
    inspect an individual execution. Every message and measurement is synthetic.
    """)
    return


@app.cell
def _(load_spans):
    spans = load_spans("data/agent-traces.json")
    return (spans,)


@app.cell
def _(mo, spans):
    service_filter = mo.ui.dropdown(
        options=["All", *spans["service_name"].unique().sort().to_list()],
        value="All", label="Service",
    )
    service_filter
    return (service_filter,)


@app.cell
def _(
    invocation_details,
    pl,
    service_filter,
    spans,
    summarize_invocations,
    summarize_operations,
):
    selected_spans = spans.filter(
        pl.lit(True) if service_filter.value == "All"
        else pl.col("service_name") == service_filter.value
    )
    service_summary = summarize_invocations(selected_spans)
    operation_summary = summarize_operations(selected_spans)
    invocations = invocation_details(selected_spans)
    return invocations, operation_summary, selected_spans, service_summary


@app.cell
def _(mo, px, service_summary):
    mo.vstack([
        mo.md("## Agent invocations\nFailure rate is measured on root spans, not all spans."),
        mo.ui.table(service_summary),
        px.bar(service_summary, x="service_name", y="p95_duration_ms",
               color="service_name", hover_data=["invocations", "failure_rate"],
               title="P95 invocation duration (synthetic sample)"),
    ])
    return


@app.cell
def _(mo, operation_summary, px):
    mo.vstack([
        mo.md("## Where does the time go?\nThese are child operations; their errors can be recovered."),
        px.bar(operation_summary, x="operation_name", y="median_duration_ms",
               color="service_name", barmode="group", hover_data=["calls", "error_rate"],
               title="Median duration per model or tool call"),
    ])
    return


@app.cell
def _(invocations, mo):
    outcome_filter = mo.ui.dropdown(
        options=["All", *invocations["outcome"].unique().sort().to_list()],
        value="All", label="Invocation outcome",
    )
    outcome_filter
    return (outcome_filter,)


@app.cell
def _(invocations, mo, outcome_filter, pl):
    visible_invocations = invocations.filter(
        pl.lit(True) if outcome_filter.value == "All"
        else pl.col("outcome") == outcome_filter.value
    )
    mo.ui.table(visible_invocations)
    return (visible_invocations,)


@app.cell
def _(mo, visible_invocations):
    trace_ids = visible_invocations["trace_id"].to_list()
    mo.stop(not trace_ids, mo.md("No invocations for this selection."))
    trace_selector = mo.ui.dropdown(options=trace_ids, value=trace_ids[0], label="Trace")
    trace_selector
    return (trace_selector,)


@app.cell
def _(pl, selected_spans, trace_selector):
    trace = selected_spans.filter(pl.col("trace_id") == trace_selector.value).sort("start_ns")
    timeline = trace.with_columns(
        ((pl.col("start_ns") - pl.col("start_ns").min()) / 10**6).alias("offset_ms"),
        pl.Series("step", [f"{index + 1}. {name}" for index, name in enumerate(trace["span_name"])]),
    )
    return timeline, trace


@app.cell
def _(mo, px, timeline, trace):
    mo.vstack([
        mo.md("## One execution\nThe root encloses its children: overlapping bars are expected."),
        px.bar(timeline, x="duration_ms", y="step", base="offset_ms", orientation="h",
               color="status_code", hover_data=["span_id", "operation_name"],
               title="Span timeline, milliseconds since invocation start"),
        mo.ui.table(trace.select("span_name", "status_code", "attributes_json")),
    ])
    return


if __name__ == "__main__":
    app.run()
