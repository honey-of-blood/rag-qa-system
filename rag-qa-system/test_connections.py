import uuid
import sys
import os
from frontend import api_health, api_documents, api_query, api_upload

print("Testing health endpoint...")
health = api_health()
if "error" in health:
    print(f"Health check failed: {health['error']}")
    sys.exit(1)
print(f"Health response: {health}")

print("\nTesting upload endpoint...")
# Create a dummy pdf
with open("dummy.pdf", "wb") as f:
    f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF")
    
class MockFile:
    def __init__(self, name):
        self.name = name

upload_res = api_upload([MockFile("dummy.pdf")])
if "error" in upload_res:
    print(f"Upload check failed: {upload_res['error']}")
    sys.exit(1)
print(f"Upload response: {upload_res}")

print("\nTesting documents endpoint...")
docs = api_documents()
if "error" in docs:
    print(f"Documents check failed: {docs['error']}")
    sys.exit(1)
print(f"Documents response: {docs.get('documents', [])}")

print("\nTesting query endpoint...")
session_id = str(uuid.uuid4())
query_res = api_query("What does the uploaded PDF say?", session_id)
if "error" in query_res:
    print(f"Query check failed: {query_res['error']}")
    sys.exit(1)
print(f"Query response answer: {query_res.get('answer', '')}")

print("\nAll backend-frontend connection APIs tested successfully!")
