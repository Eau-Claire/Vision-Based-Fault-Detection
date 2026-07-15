import sys
import requests
import io
import time
import jwt

BASE_URL = "http://localhost:5194"
SERVICE_KEY = "Reme8lqiErnO9ZppU0SeNattf4ObRvbv"
JWT_SECRET = "MzUfGom64CuJiOwIoGB64kWDwJ3nG7yZB4wYbqKFszT"

# Minimal 1x1 black JPEG
DUMMY_JPEG = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06'
    b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
    b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00'
    b'\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00'
    b'\x00?\x00\xbf\x00\xff\xd9'
)

# Dynamically generate token valid for 1 day
payload = {
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier": "469bfac4-8b96-4f27-a772-945cff2fbaa8",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "minhchau",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "phamhoangminhchau1973@gmail.com",
    "fullname": "Minh Châu",
    "http://schemas.microsoft.com/ws/2008/06/identity/claims/role": "SystemAdmin",
    "exp": int(time.time()) + 86400,
    "iss": "UavPms.Api",
    "aud": "UavPms.Client"
}
TOKEN = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def run_e2e_flow():
    print("=== STARTING END-TO-END WORKFLOW INTEGRATION TEST ===")
    
    # Use hardcoded token
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    print("✅ Authorization header configured with active token.")

    # 2. Create Mission
    print("\n[Step 2] Creating a new inspection mission...")
    create_mission_url = f"{BASE_URL}/api/v1/missions"
    mission_payload = {
        "title": "Post-flight Drone Inspection MS-E2E",
        "routeData": "Line-A-Towers-1-5",
        "assignedToUserId": "469bfac4-8b96-4f27-a772-945cff2fbaa8",
        "droneCode": "UAV001",
        "status": "In Progress",
        "description": "E2E verification of AI workflow and completion evidence"
    }
    r = requests.post(create_mission_url, json=mission_payload, headers=headers)
    if r.status_code != 200:
        print(f"❌ Create mission failed: {r.status_code} - {r.text}")
        sys.exit(1)
        
    mission_response = r.json()
    mission_id = mission_response.get("data", {}).get("id")
    print(f"✅ Mission created successfully. MissionId: {mission_id}")

    # 3. Upload Media for AI Analysis
    print("\n[Step 3] Uploading post-flight drone image for AI analysis...")
    upload_ai_url = f"{BASE_URL}/api/v1/missions/{mission_id}/ai-analysis"
    
    files = {
        "file": ("drone_flight_frame.jpg", io.BytesIO(DUMMY_JPEG), "image/jpeg")
    }
    data = {
        "assetId": "00000000-0000-0000-0000-000000000000",
        "analysisType": "DefectDetection",
        "preferredModel": "SERVER",
        "notes": "E2E automated test upload"
    }
    
    r = requests.post(upload_ai_url, files=files, data=data, headers=headers)
    if r.status_code != 200:
        print(f"❌ Upload for AI failed: {r.status_code} - {r.text}")
        sys.exit(1)
        
    upload_response = r.json()
    request_id = upload_response.get("data", {}).get("id")
    print(f"✅ AI analysis request created. RequestId: {request_id}")

    # 4. Query Media ID from inspections history
    print("\n[Step 4] Querying Media ID for the uploaded file...")
    history_url = f"{BASE_URL}/api/v1/inspections/mission/{mission_id}"
    r = requests.get(history_url, headers=headers)
    if r.status_code != 200:
        print(f"❌ Failed to fetch mission inspections history: {r.status_code} - {r.text}")
        sys.exit(1)
        
    history_response = r.json()
    items = history_response.get("data", [])
    if not items:
        print("❌ No inspection media items found for the mission.")
        sys.exit(1)
        
    # Get the media ID of the item
    media_id = items[0].get("id")
    print(f"✅ Resolved MediaId: {media_id}")

    # 5. Simulate AI Callback
    print("\n[Step 5] Simulating AI Service callback with detection results...")
    callback_url = f"{BASE_URL}/api/internal/ai-analysis/results"
    callback_headers = {
        "X-AI-Service-Key": SERVICE_KEY,
        "Content-Type": "application/json"
    }
    callback_payload = {
        "requestId": request_id,
        "mediaId": media_id,
        "status": "Completed",
        "modelName": "RF-DETR-Base",
        "modelVersion": "1.0.0",
        "processingTimeMs": 250,
        "completedAt": "2026-07-15T13:00:00Z",
        "detections": [
            {
                "categoryCode": "Corrosion",
                "confidence": 0.89,
                "boundingBox": {
                    "x": 0.25,
                    "y": 0.35,
                    "width": 0.15,
                    "height": 0.20
                }
            }
        ],
        "rawResult": "{\"detections\": [{\"bbox\": [0.25, 0.35, 0.15, 0.20], \"score\": 0.89, \"class\": \"Corrosion\"}]}"
    }
    
    r = requests.post(callback_url, json=callback_payload, headers=callback_headers)
    if r.status_code != 200:
        print(f"❌ AI Callback simulation failed: {r.status_code} - {r.text}")
        sys.exit(1)
        
    print("✅ AI Callback processed successfully.")

    # 6. Verify AI Analysis Result Details
    print("\n[Step 6] Verifying AI Analysis request status and result details...")
    analysis_detail_url = f"{BASE_URL}/api/v1/ai-analysis/{request_id}"
    r = requests.get(analysis_detail_url, headers=headers)
    if r.status_code != 200:
        print(f"❌ Failed to retrieve AI analysis details: {r.status_code} - {r.text}")
        sys.exit(1)
        
    detail_response = r.json()
    status = detail_response.get("data", {}).get("status")
    result = detail_response.get("data", {}).get("result")
    
    print(f"   Request Status: {status}")
    print(f"   Request Result JSON: {result}")
    
    if status != "Completed" or not result:
        print("❌ AI analysis request is not marked Completed or lacks result JSON.")
        sys.exit(1)
    print("✅ Verification successful. AI results saved and ready for FE.")

    # 7. Upload Completion Evidence (Inspection Evidence)
    print("\n[Step 7] Uploading inspection evidence for mission completion...")
    upload_evidence_url = f"{BASE_URL}/api/v1/inspections/upload"
    
    files_evidence = {
        "file": ("evidence_report.jpg", io.BytesIO(DUMMY_JPEG), "image/jpeg")
    }
    data_evidence = {
        "missionId": mission_id,
        "assetId": "00000000-0000-0000-0000-000000000000",
        "capturedAt": "2026-07-15T13:10:00Z"
    }
    
    r = requests.post(upload_evidence_url, files=files_evidence, data=data_evidence, headers=headers)
    if r.status_code != 200:
        print(f"❌ Uploading completion evidence failed: {r.status_code} - {r.text}")
        sys.exit(1)
        
    evidence_response = r.json()
    evidence_media_id = evidence_response.get("data", {}).get("mediaId")
    print(f"✅ Completion evidence uploaded. MediaId: {evidence_media_id}")

    # 8. Close/Complete the Mission
    print("\n[Step 8] Closing the mission (updating status to Completed)...")
    update_mission_url = f"{BASE_URL}/api/v1/missions/{mission_id}"
    update_payload = {
        "title": "Post-flight Drone Inspection MS-E2E",
        "routeData": "Line-A-Towers-1-5",
        "assignedToUserId": "469bfac4-8b96-4f27-a772-945cff2fbaa8",
        "droneCode": "UAV001",
        "status": "Completed",
        "description": "E2E verification of AI workflow and completion evidence - CLOSED"
    }
    
    r = requests.put(update_mission_url, json=update_payload, headers=headers)
    if r.status_code != 200:
        print(f"❌ Failed to close mission: {r.status_code} - {r.text}")
        sys.exit(1)
        
    print("✅ Mission closed successfully.")

    # 9. Verify Final Mission Status
    print("\n[Step 9] Verifying final closed mission details...")
    get_mission_url = f"{BASE_URL}/api/v1/missions/{mission_id}"
    r = requests.get(get_mission_url, headers=headers)
    if r.status_code != 200:
        print(f"❌ Failed to retrieve final mission details: {r.status_code} - {r.text}")
        sys.exit(1)
        
    final_response = r.json()
    final_status = final_response.get("data", {}).get("status")
    print(f"   Final Mission Status: {final_status}")
    
    if final_status != "Completed":
        print(f"❌ Mission status is '{final_status}', expected 'Completed'.")
        sys.exit(1)
        
    print("\n=== E2E WORKFLOW INTEGRATION TEST PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_e2e_flow()
