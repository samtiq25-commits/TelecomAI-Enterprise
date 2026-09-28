import sqlite3

connection = sqlite3.connect("telecomai.db")

tables = connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print("Tables:")
for table in tables:
    print(table[0])

connection.close()