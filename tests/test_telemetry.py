"""Checks for the distinction between recovered errors and agent failures."""

import json

from reproducible_notebook.telemetry import invocation_details, load_spans, summarize_invocations


def test_recovered_tool_error_does_not_count_as_agent_failure(tmp_path):
    #path = tmp_path / "traces.json"
    #path.write_text(json.dumps(generate_traces(count=15, seed=42)), encoding="utf-8")
    spans = load_spans("data/agent-traces.json")
    details = invocation_details(spans)
    recovered = details.filter(details["outcome"] == "recovered")
    failed = details.filter(details["outcome"] == "failed")

    assert recovered.height == 10
    assert recovered["tool_errors"].to_list() == [1] * 10
    assert recovered["tool_attempts"].to_list() == [2] * 10
    assert recovered["status_code"].to_list() == ["UNSET"] * 10
    assert failed.height == 6
    assert failed["tool_attempts"].to_list() == [1] * 6
    assert summarize_invocations(spans)["failure_rate"].to_list() == [0.2] * 3


def test_finished_children_fit_inside_their_root_and_calls_have_matching_ids(tmp_path):
    #payload = generate_traces(count=15, seed=42)
    #path = tmp_path / "traces.json"
    #path.write_text(json.dumps(path), encoding="utf-8")
    spans = load_spans("data/agent-traces.json")
    for group in spans.partition_by("trace_id"):
        root = group.filter(group["parent_span_id"].is_null()).row(0, named=True)
        children = group.filter(group["parent_span_id"].is_not_null())
        assert all(parent == root["span_id"] for parent in children["parent_span_id"])
        root_end = root["start_ns"] + int(root["duration_ms"] * 10**6)
        requests, responses = [], []
        for child in children.to_dicts():
            assert root["start_ns"] <= child["start_ns"]
            assert child["start_ns"] + int(child["duration_ms"] * 10**6) <= root_end
            attrs = json.loads(child["attributes_json"])
            if child["operation_name"] == "chat":
                for output in attrs["gen_ai.output.messages"]:
                    requests.extend(part["id"] for part in output["parts"] if part["type"] == "tool_call")
            else:
                responses.append(attrs["gen_ai.tool.call.id"])
        assert requests == responses
