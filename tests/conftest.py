import pytest

import patch_agent

# Patch before any test runs, so every agent call is tracked
patch_agent.install()


@pytest.fixture(autouse=True)
def track_test_cost(request):
    """Attribute the cost of each test to that test."""
    patch_agent.tracker.current_test = request.node.name
    yield
    patch_agent.tracker.current_test = None


def pytest_terminal_summary(terminalreporter):
    tracker = patch_agent.tracker
    if not tracker.by_model:
        return

    terminalreporter.section("LLM cost")
    for test, cost in sorted(tracker.by_test.items(), key=lambda x: -x[1]):
        terminalreporter.write_line(f"{test}: ${cost:.4f}")
    terminalreporter.write_line("")
    terminalreporter.write_line(tracker.report())
