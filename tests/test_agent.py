from sql_agent import sql_agent, SQLResult
from utils import collect_tools
from judge import assert_criteria

# Ground truth, found by running this query directly in DuckDB:
#   SELECT COUNT(*) FROM trips WHERE passenger_count > 5
# The data never changes, so the exact number is safe to hard-code.
EXPECTED_COUNT = 22413  # <-- replace with the number DuckDB returns


def test_trips_with_more_than_5_passengers():
    result = sql_agent.run_sync("How many trips had more than 5 passengers?")
    output = result.output

    assert isinstance(output, SQLResult)

    # The agent must report the SQL it used
    assert isinstance(output.sql_query, str)
    assert output.sql_query.strip() != ""

    # The answer must contain the real count. The model may format it as
    # "1,234" or "1 234", so remove separators before checking.
    normalized = output.result_text.replace(",", "").replace(" ", "")
    assert str(EXPECTED_COUNT) in normalized, (
        f"Expected {EXPECTED_COUNT} in result_text, got: {output.result_text!r}"
    )


def test_agent_gets_schema_before_running_sql():
    result = sql_agent.run_sync("What is the most common payment type?")

    tool_calls = collect_tools(result.new_messages())
    tool_names = [call.tool_name for call in tool_calls]
    print(tool_names)

    # The agent must look at the schema before doing anything else
    assert len(tool_names) > 0, "The agent did not call any tools"
    assert tool_names[0] == "get_schema", f"First tool call was {tool_names[0]}"

    # ... and then actually query the data
    assert "run_sql" in tool_names, f"run_sql was never called: {tool_names}"


def test_highest_average_fare_by_hour():
    result = sql_agent.run_sync(
        "Which hour of the day has the highest average fare amount?"
    )

    assert_criteria(
        result,
        [
            "the SQL query correctly calculates average fare by hour of day",
            "the result identifies a specific hour as having the highest average fare",
            "the result includes the actual average fare amount",
        ],
    )
