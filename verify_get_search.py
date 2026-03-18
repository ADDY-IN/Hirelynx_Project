import requests
import json

BASE_URL = "http://localhost:8000"
ADMIN_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjEsImVtYWlsIjoiYWRtaW4xQHlvcG1haWwuY29tIiwicm9sZSI6IkFETUlOIiwiaWF0IjoxNzczODMzODUwLCJleHAiOjE3NzQwMDY2NTB9.rpPd1pQ0dRv27cxzPZdsENsDqFYCpiL737WXcTMDDPc"

def test_get_search():
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    
    # Test 1: Empty Query
    print("\n--- Test 1: Empty Query (GET) ---")
    response = requests.get(f"{BASE_URL}/v1/admin/candidates/search?query=&limit=10", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"Results Count: {data['count']}")
        for r in data['results']:
            print(f"ID: {r['candidateId']}, Score: {r['score']}, Recommendation: {r['recommendation']}")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

    # Test 2: Search for 'Kunal'
    print("\n--- Test 2: Search for 'Kunal' (GET) ---")
    response = requests.get(f"{BASE_URL}/v1/admin/candidates/search?query=Kunal&limit=10", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"Results Count: {data['count']}")
        for r in data['results']:
            print(f"ID: {r['candidateId']}, Score: {r['score']}, Recommendation: {r['recommendation']}")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Note: This script assumes the server is running on localhost:8000
    # Since I cannot easily start the server in the background and wait for it,
    # I will also verify via a direct DB call in the same script if needed.
    test_get_search()
