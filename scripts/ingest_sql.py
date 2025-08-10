"""
Seed a local SQLite copy of Northwind if it does not exist
"""

import os, subprocess, pathlib, urllib.request, sqlite3, gzip, shutil

DB_PATH = pathlib.Path("data/northwind.db")
if DB_PATH.exists():
    print("Northwind already present")
    exit()

url = "https://github.com/microsoft/sql-server-samples/raw/master/samples/databases/northwind-pubs/instnwnd.sql.gz"
dst = "instnwnd.sql.gz"
urllib.request.urlretrieve(url, dst)
with gzip.open(dst, "rb") as gz, open("instnwnd.sql", "wb") as sql:
    shutil.copyfileobj(gz, sql)
con = sqlite3.connect(DB_PATH)
with open("instnwnd.sql") as f:
    con.executescript(f.read())
con.close()
print("Northwind created")
