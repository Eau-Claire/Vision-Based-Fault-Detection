# Vision-Based Fault Detection (UAV Power-Line Inspection Service)

Hệ thống nhận diện lỗi lưới điện 110kV sử dụng Drone AI kết hợp sức mạnh của **YOLO11** (Phát hiện khẩn cấp/biên) và **RF-DETR** (Phân tích ngoại tuyến chính xác cao).

## Kiến trúc Hệ thống (Architecture)

Dự án được triển khai dưới dạng monorepo với 2 module chạy độc lập chia sẻ chung các schemas và contract giao tiếp với backend ASP.NET Core:

1. **Edge Raspberry Module (`edge_raspberry/`)**:
   - Sử dụng **YOLO11** (CPU-only, tối ưu hóa cho ARM64).
   - Dùng để phát hiện nhanh các bất thường trong lúc bay.
   - Nhận nhiệm vụ từ queue `ai.analysis.edge.requested`.
2. **Server PC Module (`server_pc/`)**:
   - Sử dụng **RF-DETR** (Hỗ trợ GPU CUDA / fallback CPU).
   - Dùng để phân tích chi tiết hình ảnh/video độ phân giải cao sau khi drone upload.
   - Nhận nhiệm vụ từ queue `ai.analysis.server.requested`.
3. **Shared Module (`shared/`)**:
   - Chứa DTO, schemas chuẩn hóa, logic tải media, class mapping, xử lý bbox và callback retry.

---

## Cấu trúc thư mục

```text
ai-project/
├── shared/                 # Contract & utilities chung
│   ├── schemas/            # Schemas request/result
│   ├── services/           # Tải file, callback, mapping
│   ├── messaging/          # RabbitMQ client kết nối tự phục hồi
│   ├── utils/              # Bbox normalization & logging
│   └── config/             # BaseSettings
│
├── edge_raspberry/         # Module chạy trên Raspberry Pi (YOLO11)
│   ├── app/                # Main app, detector, consumer, settings
│   ├── Dockerfile          # Build đa kiến trúc (ARM64/AMD64)
│   └── requirements.txt
│
├── server_pc/              # Module chạy trên PC/Server (RF-DETR)
│   ├── app/                # Main app, detector, consumer, settings
│   ├── Dockerfile          # Hỗ trợ CUDA GPU
│   └── requirements.txt
│
└── docker-compose.yml      # Orchestration dịch vụ
```

---

## Hướng dẫn Khởi chạy Dịch vụ

### 1. Cài đặt Cấu hình Môi trường
Sao chép tệp mẫu cấu hình sang `.env` và tùy chỉnh theo nhu cầu:
```bash
cp .env.example .env
```

### 2. Khởi chạy bằng Docker Compose
Dùng Docker Compose để chạy cả 2 module AI cùng dịch vụ RabbitMQ đi kèm:
```bash
docker-compose up --build -d
```

### 3. Chạy từng Module độc lập cục bộ (Local Development)

#### Thiết lập Virtual Environment chung:
```bash
python3 -m venv .venv
source .venv/bin/python
pip install -r edge_raspberry/requirements.txt   # Cho Edge development
# HOẶC
pip install -r server_pc/requirements.txt        # Cho Server development
```

#### Khởi chạy Edge Module (YOLO11):
```bash
export DEVICE_PROFILE=edge
python3 -m edge_raspberry.app.main
```
Ứng dụng sẽ lắng nghe tại cổng `8001`.

#### Khởi chạy Server Module (RF-DETR):
```bash
export DEVICE_PROFILE=server
python3 -m server_pc.app.main
```
Dịch vụ sẽ khởi động và lắng nghe tại cổng `8002`.

---

## Kiểm thử (Testing)

Chạy bộ unit test kiểm tra bbox, class mapping, callback retry, và routing rules:
```bash
.venv/bin/python -m unittest discover -s shared/tests -p "test_*.py"
```

## Các API Endpoint chính (cả 2 Runtimes)
- `GET /health`: Kiểm tra trạng thái tiến trình (liveness).
- `GET /ready`: Kiểm tra mô hình đã load thành công chưa (readiness).
- `POST /api/analyze`: Endpoint RESTful gửi yêu cầu phân tích bất đồng bộ.