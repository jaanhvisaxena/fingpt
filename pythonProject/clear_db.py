import sqlite3

conn = sqlite3.connect("invoices.db")
c = conn.cursor()
c.execute("DELETE FROM invoice_data")
conn.commit()
conn.close()
print("All invoice data cleared.") 