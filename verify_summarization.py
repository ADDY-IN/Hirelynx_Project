import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_candidate_summarization(candidate_id):
    print(f"\n--- Testing Candidate Summarization (ID: {candidate_id}) ---")
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/v1/candidate/{candidate_id}/summarize")
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {response.status_code}")
            print(f"Time Taken: {end_time - start_time:.2f}s")
            print(f"Summary: {data.get('summary')}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_job_summarization(job_id):
    print(f"\n--- Testing Job Summarization (ID: {job_id}) ---")
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/v1/recruiter/jobs/{job_id}/summarize")
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {response.status_code}")
            print(f"Time Taken: {end_time - start_time:.2f}s")
            print(f"Summary: {data.get('summary')}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Ensure server is running before testing
    print("Testing AI Summarization Endpoints...")
    test_candidate_summarization(18)
    test_job_summarization(28)
