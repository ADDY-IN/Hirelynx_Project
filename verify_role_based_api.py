import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_role_based_api():
    print("--- 1. Testing Candidate Perspective ---")
    # Candidate 18 recommended jobs
    resp = requests.get(f"{BASE_URL}/v1/candidate/18/recommended-jobs")
    print(f"Candidate Recs: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Found {len(data)} jobs. Top Match Percentage: {data[0].get('matchPercentage')}%")

    # Candidate searches for jobs
    resp = requests.post(f"{BASE_URL}/v1/candidate/search-jobs?query=Python Developer")
    print(f"Candidate Job Search: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Found {len(resp.json())} relevant jobs")

    print("\n--- 2. Testing Recruiter Perspective ---")
    # Recruiter top candidates for Job 28
    resp = requests.get(f"{BASE_URL}/v1/recruiter/jobs/28/top-candidates")
    print(f"Recruiter Top Candidates: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Found {len(data)} candidates. Top Suitability Score: {data[0].get('suitabilityScore')}")

    # Recruiter searches for candidates
    resp = requests.post(f"{BASE_URL}/v1/recruiter/search-candidates?query=Backend Developer with AWS")
    print(f"Recruiter Candidate Search: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Found {len(resp.json())} matching candidates")

if __name__ == "__main__":
    try:
        test_role_based_api()
    except Exception as e:
        print(f"Error: {e}")
