"""
More scenario tests for the SQL agent.

Every test checks three things:
1. Tool order: get_schema is called first, then run_sql.
2. SQL: the query uses the columns the question needs.
3. Value: result_text contains the correct answer.

The correct answer is computed by running the right query directly in DuckDB
(ground truth), so the tests do not depend on hard-coded numbers.
"""

import re

from sql_agent import sql_agent
from sql_tools import con
from utils import collect_tools
from judge import assert_criteria


# ---------- helpers ----------

def query_value(sql):
    """Run a query directly in DuckDB and return the first value."""
    return con.execute(sql).fetchone()[0]


def run_agent(question):
    result = sql_agent.run_sync(question)
    output = result.output
    tool_names = [call.tool_name for call in collect_tools(result.new_messages())]

    print(f"\nQuestion:   {question}")
    print(f"Tool calls: {tool_names}")
    print(f"SQL:        {output.sql_query}")
    print(f"Answer:     {output.result_text}")

    return result, output, tool_names


def assert_schema_then_sql(tool_names):
    assert len(tool_names) > 0, "The agent did not call any tools"
    assert tool_names[0] == "get_schema", f"First tool call was {tool_names[0]}"
    assert "run_sql" in tool_names, f"run_sql was never called: {tool_names}"
    first_sql = tool_names.index("run_sql")
    assert "get_schema" in tool_names[:first_sql], "run_sql was called before get_schema"


def assert_sql_uses(sql_query, *columns):
    sql = sql_query.lower()
    for column in columns:
        assert column.lower() in sql, f"Column {column!r} not in SQL: {sql_query}"


def numbers_in(text):
    """Extract all numbers from text, e.g. '1,234 trips, $3.18' -> [1234.0, 3.18]."""
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def assert_contains_number(text, expected, tolerance=0.01):
    found = numbers_in(text)
    assert any(abs(n - expected) <= tolerance for n in found), (
        f"Expected {expected} (±{tolerance}) in result_text, found {found}: {text!r}"
    )


# ---------- tests ----------

def test_average_tip_for_credit_card():
    # Scenario: filter on a coded column (payment_type = 1 means credit card)
    expected = query_value(
        "SELECT AVG(tip_amount) FROM trips WHERE payment_type = 1"
    )

    _, output, tool_names = run_agent(
        "What is the average tip amount for credit card payments?"
    )

    assert_schema_then_sql(tool_names)
    assert_sql_uses(output.sql_query, "tip_amount", "payment_type")
    assert_contains_number(output.result_text, round(expected, 2))


def test_pickup_location_with_most_trips():
    # Scenario: GROUP BY + ORDER BY count, return an ID (not a count)
    expected = query_value("""
        SELECT PULocationID FROM trips
        GROUP BY PULocationID
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    _, output, tool_names = run_agent(
        "Which pickup location (PULocationID) has the most trips?"
    )

    assert_schema_then_sql(tool_names)
    assert_sql_uses(output.sql_query, "PULocationID", "count")
    assert_contains_number(output.result_text, expected, tolerance=0)


def test_average_fare_for_long_trips():
    # Scenario: numeric filter on one column, aggregate on another
    expected = query_value(
        "SELECT AVG(fare_amount) FROM trips WHERE trip_distance > 10"
    )

    _, output, tool_names = run_agent(
        "What is the average fare for trips longer than 10 miles?"
    )

    assert_schema_then_sql(tool_names)
    assert_sql_uses(output.sql_query, "fare_amount", "trip_distance", "10")
    assert_contains_number(output.result_text, round(expected, 2))


def test_trips_with_zero_passengers():
    # Scenario: edge case, 0 is not the same as NULL
    expected = query_value("SELECT COUNT(*) FROM trips WHERE passenger_count = 0")

    _, output, tool_names = run_agent(
        "How many trips had zero passengers recorded?"
    )

    assert_schema_then_sql(tool_names)
    assert_sql_uses(output.sql_query, "passenger_count", "0")
    # If the agent wrongly counts NULLs as zero, the number will not match
    assert_contains_number(output.result_text, expected, tolerance=0)


def test_busiest_day_of_week():
    # Scenario: derive a value from a timestamp; answer is a word, not a number
    expected_day = query_value("""
        SELECT dayname(tpep_pickup_datetime) AS day FROM trips
        GROUP BY day
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    result, output, tool_names = run_agent(
        "What is the busiest day of the week for taxi trips?"
    )

    assert_schema_then_sql(tool_names)
    assert_sql_uses(output.sql_query, "tpep_pickup_datetime")
    assert expected_day.lower() in output.result_text.lower(), (
        f"Expected {expected_day} in result_text: {output.result_text!r}"
    )

    # The SQL can be written in many ways (dayname, dayofweek, strftime),
    # so we also let the judge check it.
    assert_criteria(
        result,
        [
            "the SQL query groups trips by day of the week using the pickup datetime",
            "the result names a single day of the week as the busiest",
            "the result includes the number of trips for that day",
        ],
    )
