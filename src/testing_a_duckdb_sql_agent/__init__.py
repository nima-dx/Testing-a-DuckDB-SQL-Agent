def main() -> None:
    from sql_tools import setup_database

    count = setup_database()
    print(f"Hello from testing-a-duckdb-sql-agent! {count} rows loaded.")
