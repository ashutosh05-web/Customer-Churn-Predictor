"""
Loads the cleaned CSV into a SQLite database so we can demonstrate
SQL-based analysis (not everything needs to be pandas).
"""
import sqlite3
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_to_sql(df: pd.DataFrame, cfg: dict):
    db_path = cfg["sql"]["db_path"]
    table = cfg["sql"]["table_name"]
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into {db_path} (table: {table})")


def run_query(query: str, cfg: dict) -> pd.DataFrame:
    conn = sqlite3.connect(cfg["sql"]["db_path"])
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"])
    load_to_sql(df, cfg)
