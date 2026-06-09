from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(title="Mock Web API Backend")

@app.post("/api/faults")
async def faults(request: Request):
    try:
        data = await request.json()
        print("\n📥 [MOCK SERVER RECEIVED FAULT LOG]")
        print("--------------------------------------------------")
        print(f"Tower ID:         {data.get('towerId')}")
        print(f"Fault Type:       {data.get('faultType')}")
        print(f"Confidence Score: {data.get('confidenceScore'):.2%}" if isinstance(data.get('confidenceScore'), float) else f"Confidence Score: {data.get('confidenceScore')}")
        print(f"Image Path:       {data.get('imagePath')}")
        print(f"Location:         Latitude {data.get('latitude')}, Longitude {data.get('longitude')}")
        print("--------------------------------------------------")
        return {"status": "success", "message": "Fault reported successfully to mock backend"}
    except Exception as e:
        print(f"❌ Error parsing fault payload: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/faults/upload-image")
async def upload(request: Request):
    print("\n📥 [MOCK SERVER RECEIVED IMAGE UPLOAD]")
    # Since it's a multipart form upload, we can log the headers or content type
    print(f"Content-Type: {request.headers.get('content-type')}")
    # Return both imagePath and imageUrl to support different clients
    return {
        "imagePath": "/uploads/mock_fault_capture.jpg",
        "imageUrl": "mock_fault_capture.jpg"
    }

if __name__ == "__main__":
    print("Starting Mock Backend Server on port 5000...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
