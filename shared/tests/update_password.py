import bcrypt
import psycopg2

new_pw = b"Password@123"
hashed = bcrypt.hashpw(new_pw, bcrypt.gensalt(10)).decode('utf-8')

conn_str = "host=aws-1-ap-northeast-1.pooler.supabase.com port=5432 dbname=postgres user=postgres.hurroumcfjmzsnzovefm password=NVMAzbbuiiS3YFl2 sslmode=require"

try:
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    
    # 1. Update password
    cursor.execute(
        'UPDATE "Users" SET "PasswordHash" = %s WHERE "Email" = %s',
        (hashed, "uselessliem@gmail.com")
    )
    conn.commit()
    print("Updated password hash for uselessliem@gmail.com successfully.")
    
    # 2. Fetch roles for uselessliem@gmail.com
    cursor.execute('''
        SELECT u."Email", r."RoleName"
        FROM "Users" u
        JOIN "UserRoles" ur ON u."Id" = ur."UserId"
        JOIN "Roles" r ON ur."RoleId" = r."Id"
        WHERE u."Email" = 'uselessliem@gmail.com'
    ''')
    for row in cursor.fetchall():
        print("Role:", row)
        
    cursor.close()
    conn.close()
except Exception as e:
    print("Error:", e)
