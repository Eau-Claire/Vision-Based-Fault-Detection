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
   - Chứa client Roboflow Workflow dùng `inference-sdk` để gọi workflow hosted khi cần tích hợp inference qua Roboflow Serverless.

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

#### Khởi chạy Server Module (Harness mặc định, Roboflow/RF-DETR optional):
```bash
export DEVICE_PROFILE=server
export SERVER_INFERENCE_BACKEND=harness
python3 -m server_pc.app.main
```
Dịch vụ sẽ khởi động và lắng nghe tại cổng `8002`. Để dùng Roboflow hosted Workflow, đặt `SERVER_INFERENCE_BACKEND=roboflow` và cấu hình `ROBOFLOW_API_KEY`. Để quay lại RF-DETR local, đặt `SERVER_INFERENCE_BACKEND=local`.

Docker `server_pc` mặc định dùng image CPU/harness nhẹ và không cài RF-DETR/PyTorch CUDA. Deploy riêng server PC bằng:

```bash
docker compose -f docker-compose.prod.yml up -d --build server_pc
```

Chỉ khi cần backend RF-DETR local mới build thêm dependency ML CPU-only:

```bash
INSTALL_LOCAL_ML=true SERVER_INFERENCE_BACKEND=local docker compose -f docker-compose.prod.yml up -d --build server_pc
```

---

## Roboflow Workflow Integration

Workflow hosted đã được tích hợp trong `shared/services/roboflow_workflow_client.py`. `server_pc` hiện dùng `HarnessRuntime` mặc định để có flow provider-independent ổn định; Roboflow hosted Workflow là backend optional qua `SERVER_INFERENCE_BACKEND=roboflow`.

- Workspace slug: `les-workspace-ijdwd`
- Workflow slug: `evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic`
- API URL: `https://serverless.roboflow.com`
- Declared input: `image`
- Declared runtime parameters: none
- Declared output key: `predictions`

Thiết lập API key qua biến môi trường, không commit secret:
```bash
export ROBOFLOW_API_KEY="<key-from-app.roboflow.com/settings/api>"
```

Ví dụ gọi workflow cho một ảnh HTTPS:
```python
from shared.services.roboflow_workflow_client import run_evn_object_detection_workflow

runs = run_evn_object_detection_workflow(
    "https://media.roboflow.com/notebooks/examples/dog.jpeg",
)
detection_result = runs[0].as_detection_result()
```

Client dùng `InferenceHTTPClient.run_workflow(...)`, timeout theo attempt, retry với exponential backoff, và parse output theo key thực tế của workflow. Nếu workflow về sau thêm output dạng ảnh base64, truyền `output_dir` để decode ra file thay vì log payload base64.

Lưu ý: lần kiểm tra live ngày 2026-07-19 bằng Roboflow MCP trả về lỗi server-side 500 vì workflow wrapper đang bind `model_id` vào child workflow, trong khi child workflow chỉ khai báo `class_agnostic_nms`, `confidence`, `image`, `iou_threshold`, và `max_detections`. Cần publish lại workflow trên Roboflow trước khi smoke test live có thể pass.

## Kiểm thử (Testing)

Chạy bộ unit test kiểm tra bbox, class mapping, callback retry, và routing rules:
```bash
.venv/bin/python -m unittest discover -s shared/tests -p "test_*.py"
```

Chạy smoke test live Roboflow sau khi workflow published hoạt động:
```bash
export ROBOFLOW_API_KEY="<key-from-app.roboflow.com/settings/api>"
export RUN_ROBOFLOW_SMOKE_TEST=1
.venv/bin/python -m unittest shared.tests.test_roboflow_workflow_client.TestRoboflowWorkflowSmoke
```

## Các API Endpoint chính (cả 2 Runtimes)
- `GET /health`: Kiểm tra trạng thái tiến trình (liveness).
- `GET /ready`: Kiểm tra mô hình đã load thành công chưa (readiness).
- `POST /api/analyze`: Endpoint RESTful gửi yêu cầu phân tích bất đồng bộ.