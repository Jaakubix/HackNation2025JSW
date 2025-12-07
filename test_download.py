import requests
import sys

API_URL = "http://127.0.0.1:8000"
# Use the filename that we know exists and has data
FILENAME = "processed_nagranie3_long_test.mp4" 

def test_download():
    print(f"Testing download for {FILENAME}...")
    
    url = f"{API_URL}/api/report/{FILENAME}"
    print(f"Requesting: {url}")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            print(f"Content Length: {len(content)}")
            print("--- Content Start ---")
            print(content[:500])
            print("--- Content End ---")
            
            if len(content.strip()) == 0:
                print("FAILURE: Received empty content!")
            else:
                print("SUCCESS: Received data.")
        else:
            print(f"FAILURE: Server returned error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_download()
