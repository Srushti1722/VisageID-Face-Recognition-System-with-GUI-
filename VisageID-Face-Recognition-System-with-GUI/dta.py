import sqlite3
import sys

# Set UTF-8 encoding for stdout (fixes emoji printing issue)
sys.stdout.reconfigure(encoding='utf-8')

def clear_database():
    conn = sqlite3.connect("database.db")  # Connect to the database
    c = conn.cursor()

    # Get all user-defined table names (excluding system tables)
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = c.fetchall()

    # Delete data from each table
    for table in tables:
        table_name = table[0]
        c.execute(f"DELETE FROM {table_name};")  # Clear table
        c.execute(f"DELETE FROM sqlite_sequence WHERE name = '{table_name}';")  # Reset autoincrement
        print(f"✅ Deleted all data from {table_name}")

    conn.commit()
    conn.close()
    print("✅ Database cleared successfully!")

# Run the function
clear_database()
