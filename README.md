# Vision-Based Fault Detection (UAV Inspection Client)

Hệ thống nhận diện lỗi lưới điện 110kV sử dụng Drone AI kết hợp sức mạnh của **YOLOv8** (Phát hiện vật thể) và **Custom CNN** (Phân loại chi tiết).

## Luồng hoạt động (Workflow)
```text
📱 Camera Drone/Phone → Gateway Client (Python)  
→ YOLOv8 (Tìm vị trí Sứ - trên PC)  
→ Tự động Crop vùng ảnh Sứ  
→ Custom CNN (Soi lỗi Sạch/Bẩn)  
→ Gửi HTTP POST báo cáo lỗi trực tiếp về .NET Web API
```

## Cấu trúc dự án
- `/training`: Chứa mã nguồn huấn luyện mô hình (YOLOv8 và Custom CNN MobileNetV3).
- `/inference`:
  - `gateway.py`: File khởi chạy client chính. Đọc luồng video, chạy AI và gửi kết quả về .NET API.
  - `config.py`: File cấu hình tập trung (hỗ trợ profiles `pc` hoặc `rasp`).
- `/models`: Lưu trữ các file trọng số model (`.pt` và `.onnx`).

## Hướng dẫn khởi chạy

### 1. Cấu hình môi trường (.env)
Sao chép `.env.example` thành `.env` và thiết lập profile phù hợp:
```env
DEVICE_PROFILE=pc           # Hoặc "rasp" nếu chạy trên Raspberry Pi
IP_CAMERA_URL=0             # 0 là webcam, hoặc đường dẫn camera IP
OCELOT_GATEWAY_URL=http://localhost:5000 # Đường dẫn của .NET Ocelot Gateway
```

### 2. Cài đặt thư viện
```bash
pip install opencv-python ultralytics torch torchvision pillow python-dotenv requests
```

### 3. Chạy hệ thống client
```bash
cd inference
python gateway.py
```
Ứng dụng sẽ mở cửa sổ hiển thị camera trực tiếp với các bounding boxes được vẽ theo thời gian thực và tự động báo cáo lỗi về server .NET.