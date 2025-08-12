import os
import pyodbc

# MSSQL 連線參數，建議用環境變數管理比較安全
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "140.118.115.196")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "Northwind")
MSSQL_USER = os.getenv("MSSQL_USER", "llm")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "1qaz2WSX")

conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={MSSQL_SERVER};"
    f"DATABASE={MSSQL_DATABASE};"
    f"UID={MSSQL_USER};"
    f"PWD={MSSQL_PASSWORD};"
    f"TrustServerCertificate=yes;"
)

def main():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # 1. 取得所有表格名稱
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        print(f"資料庫 '{MSSQL_DATABASE}' 中的表格：")
        for t in tables:
            print(f"- {t}")

        print("\n每個表格的欄位資訊：")
        # 2. 依序列出每個表的欄位與型態
        for table in tables:
            print(f"\n表格: {table}")
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table}'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()
            for col in columns:
                col_name, data_type, max_len = col
                length_info = f"({max_len})" if max_len else ""
                print(f"  - {col_name}: {data_type}{length_info}")

        cursor.close()
        conn.close()

    except Exception as e:
        print("連線或查詢發生錯誤：", e)

if __name__ == "__main__":
    main()
