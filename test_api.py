import sys
import logging
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def run_api_tests():
    print("Testing /")
    response = client.get("/")
    assert response.status_code == 200
    print("PASS: /")

    # Mocks or invalid S3 keys should throw 400s cleanly without crashing the server
    print("Testing /v1/parser/resume with invalid key")
    response = client.post("/v1/parser/resume?s3_key=fake/path/resume.pdf")
    assert response.status_code == 400
    print("PASS: /v1/parser/resume error handling")
    
    print("Testing /v1/scoring/match-all")
    response = client.post("/v1/scoring/match-all")
    assert response.status_code == 200
    print("PASS: /v1/scoring/match-all triggers background task")

if __name__ == "__main__":
    run_api_tests()
    print("All API structural tests passed!")
