from agent.log_graph import LogClassification, route_log


def test_missing_classification_is_sent_to_the_classifier() -> None:
    assert route_log({"log": "INFO scheduler: job completed"}) == "classify_log"


def test_restored_json_classification_is_routed() -> None:
    state = {
        "classification": {
            "severity": "warning",
            "category": "application",
            "summary": "Slow request",
        }
    }

    assert route_log(state) == "create_ticket"


def test_critical_logs_request_approval() -> None:
    state = {
        "classification": LogClassification(
            severity="critical", category="infrastructure", summary="Database unavailable"
        )
    }

    assert route_log(state) == "request_approval"


def test_warning_logs_create_a_ticket() -> None:
    state = {
        "classification": LogClassification(
            severity="warning", category="application", summary="Slow request"
        )
    }

    assert route_log(state) == "create_ticket"


def test_info_logs_are_archived() -> None:
    state = {
        "classification": LogClassification(
            severity="info", category="security", summary="Successful login"
        )
    }

    assert route_log(state) == "archive"
