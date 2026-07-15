import psycopg2

conn_str = "host=aws-1-ap-northeast-1.pooler.supabase.com port=5432 dbname=postgres user=postgres.hurroumcfjmzsnzovefm password=NVMAzbbuiiS3YFl2 sslmode=require"

try:
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u."Username", r."RoleName"
        FROM "Users" u
        JOIN "UserRoles" ur ON u."Id" = ur."UserId"
        JOIN "Roles" r ON ur."RoleId" = r."Id"
    ''')
    for row in cursor.fetchall():
        print(row)
    cursor.close()
    conn.close()
except Exception as e:
    print("Error:", e)
