import hashlib
import base64

target_b64 = "rTCOlsVpTIhDNngMhyWyXf7f2Zqj09Xx+ynvXeHWo5s="

# Brute force 100000 to 999999
for i in range(100000, 1000000):
    code_str = str(i)
    # SHA256
    h = hashlib.sha256(code_str.encode('utf-8')).digest()
    b64 = base64.b64encode(h).decode('utf-8')
    if b64 == target_b64:
        print(f"FOUND! OTP is: {code_str}")
        break
else:
    print("Not found.")
