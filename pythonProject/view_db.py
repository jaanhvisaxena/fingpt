import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("invoices.db")

# Read all data from the invoice_data table
try:
    df = pd.read_sql("SELECT * FROM invoice_data", conn)
    print(df)
except Exception as e:
    print("Error reading from database:", e)
finally:
    conn.close() 