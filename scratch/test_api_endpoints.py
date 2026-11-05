import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

endpoints = [
    ("/api/health", "GET"),
    ("/api/dashboard", "GET"),
    ("/api/model/status", "GET"),
    ("/api/model/metrics", "GET"),
    ("/api/dataset/summary", "GET"),
    ("/api/history", "GET")
]

print("="*60)
print("TESTING FASTAPI BACKEND ENDPOINTS")
print("="*60)

all_passed = True
for path, method in endpoints:
    res = client.get(path)
    status = res.status_code
    print(f"Endpoint: {path} -> HTTP {status}")
    if status != 200:
        all_passed = False
        print(f"  [ERROR] Response: {res.text}")
    else:
        sample = str(res.json())
        if len(sample) > 100:
            sample = sample[:100] + "..."
        print(f"  [OK] Data: {sample}")

print("="*60)
print("ALL ENDPOINTS VERIFIED!" if all_passed else "SOME ENDPOINTS FAILED")
print("="*60)
