# Harness Architecture Audit

## Scope and Repository Facts

This audit was written before structural harness changes. The repository in this checkout differs from some provided facts:

- No `docs/` directory existed before this audit was created.
- No `AGENTS.md` exists inside this repository. A sibling repository has `../Vision-Base-Human-Motion-Detection/docs/AGENTS.md`, but it does not apply to this checkout.
- No `alerts/` directory exists. Representative images are available under `dataset/` instead.
- There is no `edge/config.py` or `edge/clients/` package. Current edge code lives under `edge_raspberry/`; generic inference helpers live under `inference/`; shared clients/services live under `shared/`.
- The worktree already contained uncommitted Roboflow-server integration files from earlier work. They are treated as existing work and preserved.

## Current Directory Responsibilities

- `edge_raspberry/`: FastAPI service, RabbitMQ consumer, YOLO11 detector, video processor, and edge-specific settings.
- `server_pc/`: FastAPI service, RabbitMQ consumer, RF-DETR detector, video processor, server-specific settings, and a Roboflow detector added before this audit.
- `shared/`: Pydantic schemas, callback service, media downloader, RabbitMQ client, class mapping, bbox utilities, logging, and tests.
- `inference/`: Older/local gateway code with dataclass-based config and OpenCV YOLO/CNN loop logic.
- `training/`: Training and validation scripts for model development.
- `dataset/`: Local image dataset suitable for offline examples/fixtures.
- `models/`: Local model artifacts mounted into containers.
- `docker-compose*.yml`: RabbitMQ plus edge/server service orchestration.
- `test_e2e_flow.py`: External backend integration script requiring secrets and a running PMS API.

## Current Execution Flow

### Edge runtime

1. `edge_raspberry/app/main.py` loads `EdgeSettings` from environment.
2. A background thread initializes `EdgeYoloDetector`.
3. RabbitMQ consumer starts if configured.
4. Consumer downloads media using `shared.services.media_downloader.download_media`.
5. Images are decoded with OpenCV; videos are written to temporary files.
6. Detector returns `DetectionResult`.
7. Result is mapped to `AnalysisResult` and sent to callback API.
8. RabbitMQ ack/nack is coupled to callback delivery.

### Server runtime

1. `server_pc/app/main.py` loads `ServerSettings`.
2. Server detector initialization selects Roboflow or local RF-DETR based on settings introduced before this audit.
3. RabbitMQ and REST `/api/analyze` paths both download media, run detector, map result, and callback.
4. Video processing saves temp files and calls `detect_video`.

### Legacy local gateway

`inference/gateway.py` opens a camera/video source, runs YOLO/CNN in a direct while-loop, draws UI overlays, uploads fault images, and posts fault details. It mixes runtime loop, model inference, smoothing, visualization, and HTTP reporting.

## Existing External Integrations

- RabbitMQ via `pika` in `shared/messaging/rabbitmq_client.py`.
- Backend callbacks and file downloads via `requests`.
- Roboflow Serverless via `inference-sdk` in the previously added `shared/services/roboflow_workflow_client.py`.
- Local model runtimes via `ultralytics`, `rfdetr`, `transformers`, `torch`, OpenCV, and PIL.

## Coupling and Hidden Dependencies

- Consumers couple queue acknowledgement, download, inference, callback, and error policy in one function.
- Detectors are concrete classes directly initialized by app entry points; provider choice is not yet an application-level interface.
- Retry behavior exists in callback and RabbitMQ reconnect code, but not as a reusable harness policy.
- No run ID or iteration ID exists across the whole execution path; correlation IDs are request-scoped but not harness-scoped.
- Checkpoint/resume semantics do not exist. A process crash loses action history.
- `inference/gateway.py` uses global config and global cooldown state.
- Settings are split between Pydantic settings in runtime services and dataclass config in `inference/config.py`.
- Some tests/scripts under `shared/tests` are operational/security experiments (`brute_otp.py`, `query_users.py`, `crack_pw.py`, `update_password.py`) rather than unit tests.

## Missing Abstractions

- Harness runtime for run lifecycle, deadlines, cancellation, retries, checkpoints, and final result production.
- Explicit loop state machine for planning, action selection, execution, verification, memory update, and stop/escalation conditions.
- Typed per-iteration context assembly.
- Prompt builder boundary separate from runtime/tool/provider concerns.
- Provider-neutral vision workflow interface.
- Tool metadata and result taxonomy.
- Durable checkpoint store with redaction and schema versioning.
- Event stream with structured run/iteration/action/tool fields.
- Error classification across external provider failures, invalid input, missing capabilities, timeouts, and verification failures.

## Failure and Recovery Weaknesses

- Broad `except Exception` blocks classify many failures as generic inference or consumer errors.
- RabbitMQ reconnect loop retries forever without a harness-level budget or escalation condition.
- Inference failures and callback failures are handled locally but not checkpointed.
- No duplicate-execution guard for non-idempotent actions.
- Timeout handling is inconsistent across local model execution, downloads, callbacks, and external providers.
- Current Roboflow live workflow is known to fail server-side because the wrapper binds `model_id` to a child workflow that does not declare that input. The generic harness must not depend on that being fixed.

## Testability Weaknesses

- Consumers require RabbitMQ-like callback channels to test end to end.
- Local model detectors need heavy/native dependencies and model files.
- E2E script requires external backend secrets and live services.
- There is no fake provider boundary for deterministic offline tests.
- No tests cover checkpoint/resume, state transitions, redaction, duplicate prevention, or retry exhaustion.

## Security and Configuration Risks

- `.env` exists locally and must not be read into docs/checkpoints/logs.
- Service keys and Roboflow API keys are environment-based but checkpointing does not yet exist to enforce redaction.
- Callback URL restriction is present and reusable, but some paths pass broad private-IP allowances for local Docker.
- Structured logs can include free-form exception messages that may contain URLs or provider payload fragments.
- Docker Compose currently does not pass Roboflow-specific env vars to `server_pc` despite server-first Roboflow integration being present.

## Components Suitable for Reuse

- `shared.schemas.analysis_request` and `shared.schemas.analysis_result` define stable request/result DTOs.
- `shared.services.media_downloader` already handles URL resolution, max-size checks, and SSRF validation.
- `shared.services.callback_service` provides retry/backoff behavior that can be wrapped behind harness retry policy later.
- `shared.utils.bbox` and `shared.services.class_mapping` are reusable domain helpers.
- `shared.utils.logging` provides structured JSON output and correlation context.
- Existing detector protocol in `shared.schemas.detector_interface` is a useful compatibility concept for old runtimes.
- Local dataset images can serve as offline fixture inputs when `alerts/` is absent.

## Audit Conclusion

The repository is a working inference-service monorepo, not yet a harness-engineering system. The safest refactor is incremental: add a provider-independent harness/loop/context/tool/provider/memory vertical slice in a new package, keep current services running through compatibility boundaries, and migrate existing detectors/clients behind provider/tool interfaces over time.
