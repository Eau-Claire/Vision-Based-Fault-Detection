import bcrypt

pw_hash = b"$2a$10$r7v5ZIuuTlwYj.GvFazno.jriyXu8xwVJtimsx7jUnx4nsadt0BuS"

candidates = [
    "SE193418",
    "se193418",
    "Chau@123",
    "Chau12345",
    "minhchau",
    "minhchau123",
    "Minhchau123",
    "Minhchau@123",
    "password123",
    "Password123",
    "Password@123",
    "123456",
    "12345678",
    "12345678aA@",
]

for cand in candidates:
    cand_bytes = cand.encode('utf-8')
    if bcrypt.checkpw(cand_bytes, pw_hash):
        print(f"FOUND! Password is: {cand}")
        break
else:
    print("Not found in candidates list.")
