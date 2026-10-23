from pydantic import BaseModel, Field
from pydantic_ai import Agent

from sql_tools import SQLTools, con

from dotenv import load_dotenv

load_dotenv()

class SQLResult(BaseModel):
    sql_query: str = Field(description="The final SQL query used to answer the question")
    result_text: str = Field(description="The answer to the question in plain language, based on the query results")
    row_count: int = Field(description="Number of rows returned by the final query")


instructions = """
You are a data analyst who answers questions about NYC Yellow Taxi trips
from January 2024. The data is in a DuckDB table called `trips`.

Rules:
- Always start by calling get_schema to see the available columns and their types,
  before you run any query.
- Then use run_sql to answer the question. Use DuckDB SQL syntax.
- Only use columns that exist in the schema.
- If a query returns an error, read the error, fix the query and try again.
- Base your answer only on the query results. Do not guess numbers.
- Always include the actual values (counts, averages, IDs, names) in result_text.
  Round money and distances to 2 decimals.

Code values in the data:
- payment_type: 0 = Flex Fare, 1 = Credit card, 2 = Cash, 3 = No charge,
  4 = Dispute, 5 = Unknown, 6 = Voided trip
- PULocationID / DOLocationID: TLC taxi zone IDs for pickup / drop-off
- passenger_count can be NULL, which means the count was not recorded.
  When a question asks about zero (0) passengers, filter with passenger_count = 0.
  Use IS NULL only when the question asks about missing or unknown values.
- In the output, put the final SQL query you used in sql_query, a short answer
  in result_text, and the number of rows the final query returned in row_count.
""".strip()

sql_tools = SQLTools(con)

sql_agent = Agent(
    name="sql_agent",
    model="openai:gpt-4o-mini",
    instructions=instructions,
    tools=[sql_tools.get_schema, sql_tools.run_sql],
    output_type=SQLResult,
)


if __name__ == "__main__":
    question = "What's the average trip distance for rides with 2 passengers?"
    print(f"Question: {question}\n")

    result = sql_agent.run_sync(question)

    # Show which tools the agent called, in order
    for message in result.all_messages():
        for part in message.parts:
            if part.part_kind == "tool-call":
                print(f"Tool call: {part.tool_name}({part.args})")

    output = result.output
    print()
    print(f"SQL:       {output.sql_query}")
    print(f"Answer:    {output.result_text}")
    print(f"Row count: {output.row_count}")
