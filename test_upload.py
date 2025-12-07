import requests
import time
import os

API_URL = "http://127.0.0.1:8000"
FILE_PATH = "nagranie2_test.mp4"

def test_upload_and_process():
    print("Testing upload and processing...")
    
    # 1. Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return

    # 2. Upload file
    print("Testing upload and processing...")
    
    filename = "nagranie2_test.mp4"
    files = {'file': open(filename, 'rb')}
    data = {'seams': 1}
    
    response = requests.post(f"{API_URL}/api/upload", files=files, data=data)
    
    if response.status_code == 200:
        print("Upload successful:", response.json())
    else:
        print("Upload failed:", response.text)
        return

    # 4. Wait for processing using progress endpoint
    print("Waiting for processing...")
    start_time = time.time()
    max_wait = 300  # 5 minutes timeout
    
    while time.time() - start_time < max_wait:
        try:
            res = requests.get(f"{API_URL}/api/progress/{filename}")
            if res.status_code == 200:
                data = res.json()
                progress = data.get("progress", 0)
                status = data.get("status")
                print(f"Progress: {progress}% ({status})", end="\r")
                
                if status == "completed" or progress >= 100:
                    print("\nProcessing completed!")
                    break
            else:
                print(f"\nError checking progress: {res.status_code}")
        except Exception as e:
            print(f"\nConnection error: {e}")
            
        time.sleep(2)
    else:
        print("\nTimeout waiting for processing")

    # 4. Check stats
    response = requests.get(f"{API_URL}/api/stats")
    if response.status_code == 200:
        print("Stats:", response.json())
    else:
        print("Failed to get stats")

    # 5. Check for processed video
    processed_path = "processed_videos/processed_nagranie2_test.mp4"
    if os.path.exists(processed_path):
        print(f"SUCCESS: Processed video found at {processed_path}")
    else:
        print(f"FAILURE: Processed video NOT found at {processed_path}")

    # 6. Check for specific CSV report
    csv_path = "processed_videos/processed_nagranie2_test.csv"
    if os.path.exists(csv_path):
        print(f"SUCCESS: Specific CSV report found at {csv_path}")
    else:
        print(f"FAILURE: Specific CSV report NOT found at {csv_path}")

if __name__ == "__main__":
    test_upload_and_process()
