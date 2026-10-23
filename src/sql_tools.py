import os
import urllib.request

import duckdb

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"

con = duckdb.connect("taxi.db")


def setup_database():
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count


class SQLTools:
    """Tools the agent uses to explore and query the trips table."""

    def __init__(self, con, max_rows: int = 50):
        self.con = con
        self.max_rows = max_rows

    def get_schema(self) -> str:
        """
        Get the schema of the `trips` table (NYC Yellow Taxi, January 2024).

        Returns:
            One line per column in the form "column_name: TYPE".
        """
        rows = self.con.execute("DESCRIBE trips").fetchall()
        # DESCRIBE returns: column_name, column_type, null, key, default, extra
        lines = [f"{name}: {col_type}" for name, col_type, *_ in rows]
        return "Table: trips\n" + "\n".join(lines)

    def run_sql(self, query: str) -> str:
        """
        Run a DuckDB SQL query against the `trips` table and return the results.

        Args:
            query: The SQL query to execute.

        Returns:
            The column headers and data rows as text (at most 50 rows),
            or an error message if the query fails.
        """
        try:
            cursor = self.con.execute(query)
        except Exception as e:
            return f"Error: {e}"

        if cursor.description is None:
            return "Query executed. No rows returned."

        headers = [col[0] for col in cursor.description]
        # Fetch one extra row so we know whether the output was truncated
        rows = cursor.fetchmany(self.max_rows + 1)
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]

        lines = [" | ".join(headers)]
        for row in rows:
            lines.append(" | ".join(str(value) for value in row))

        if truncated:
            lines.append(f"... (showing first {self.max_rows} rows)")
        else:
            lines.append(f"({len(rows)} rows)")

        return "\n".join(lines)


if __name__ == "__main__":
    setup_database()
