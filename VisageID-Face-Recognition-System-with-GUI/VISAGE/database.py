import sqlite3
import pandas as pd
import os

# Database filename
DB_FILE = 'database.db'

# Ensure database connection is managed properly
def get_db_connection():
    return sqlite3.connect(DB_FILE)

# Convert an image to binary data
def convert_to_binary(filename):
    if not os.path.exists(filename):
        print(f"[ERROR] File '{filename}' not found.")
        return None
    with open(filename, 'rb') as file:
        return file.read()

# Insert an image into the database
def insert_image(ids, photo_path, status):
    try:
        photo_binary = convert_to_binary(photo_path)
        if photo_binary is None:
            return

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO Images (id, photo, status) VALUES (?, ?, ?)", 
                      (ids, photo_binary, status))
            conn.commit()
            print("[INFO] Image inserted successfully.")
    except sqlite3.Error as error:
        print("[ERROR] Failed to insert image:", error)

# Insert a user log entry
def insert_user_log(ids, name, date, time):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO userlog (id, name, date, time) VALUES (?, ?, ? ,?)",
                      (ids, name, date, time))
            conn.commit()
            print("[INFO] User log entry added.")
    except sqlite3.Error as error:
        print("[ERROR] Failed to insert user log:", error)

# Insert a user into userinfo table
def insert_user_info(ids, name):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO userinfo (id, name) VALUES (?, ?)", (ids, name))
            conn.commit()
            print("[INFO] User info inserted successfully.")
    except sqlite3.Error as error:
        print("[ERROR] Failed to insert user info:", error)

# Retrieve user logs
def select_user_log():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM userlog")
            records = c.fetchall()
            col_names = [desc[0] for desc in c.description]  # Get column names
            df = pd.DataFrame(records, columns=col_names)
            print(df)
            return df
    except sqlite3.Error as error:
        print("[ERROR] Failed to fetch user logs:", error)
        return pd.DataFrame()

# Retrieve user info
def select_user_info():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM userinfo")
            records = c.fetchall()
            col_names = [desc[0] for desc in c.description]  # Get column names
            df = pd.DataFrame(records, columns=col_names)
            print(df)
            return df
    except sqlite3.Error as error:
        print("[ERROR] Failed to fetch user info:", error)
        return pd.DataFrame()

# Ensure tables exist before inserting data
def create_tables():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS userlog (
                    id TEXT,
                    name TEXT,
                    date TEXT,
                    time TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS userinfo (
                    id TEXT PRIMARY KEY,
                    name TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS Images (
                    id TEXT PRIMARY KEY,
                    photo BLOB,
                    status TEXT
                )
            """)
            conn.commit()
            print("[INFO] Tables ensured.")
    except sqlite3.Error as error:
        print("[ERROR] Failed to create tables:", error)

# Ensure tables exist at the start
create_tables()
