"""
RabbitMQ consumer for server_pc runtime.

Consumes analysis jobs from the server queue, runs the configured server analysis runner,
and sends results back via callback. Acknowledges messages only
after successful callback delivery.
"""

import json
import time
import threading
from shared.schemas.analysis_request import AnalysisRequest, PreferredModel
from shared.services.callback_service import send_callback, CallbackError
from shared.services.media_downloader import download_media, DownloadError
from shared.services.result_mapper import map_success_result, map_failure_result
from shared.messaging.rabbitmq_client import consume_with_reconnect
from shared.utils.logging import (
    get_logger,
    set_correlation_context,
    clear_correlation_context,
)

logger = get_logger("server_consumer")


def create_server_consumer(analysis_runner, settings):
    """Create a RabbitMQ message handler for the server runtime.

    Args:
        analysis_runner: Initialized server analysis runner.
        settings: ServerSettings instance.

    Returns:
        Message callback function.
    """

    def on_message(ch, method, properties, body):
        """Handle incoming analysis request messages."""
        start_time = time.monotonic()

        try:
            payload = json.loads(body.decode("utf-8"))
            request = AnalysisRequest(**payload)

            # Set logging context
            set_correlation_context(
                correlation_id=request.correlation_id,
                request_id=request.request_id,
            )

            logger.info(
                f"Received server analysis job: {request.request_id}",
                extra={"event": "job_received"},
            )

            # Validate routing — reject edge-targeted jobs
            if request.preferred_model and PreferredModel.is_edge(
                request.preferred_model
            ):
                logger.warning(
                    f"Job {request.request_id} has preferredModel="
                    f"{request.preferred_model}, rejecting from server queue",
                    extra={"event": "job_rejected_wrong_runtime"},
                )
                # Reject — do not requeue (DLQ will catch it)
                ch.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )
                return

            # Download media
            try:
                file_bytes, ext = download_media(
                    file_url=request.file_url,
                    base_url=settings.callback_base_url,
                    timeout=settings.media_download_timeout,
                    max_size_bytes=settings.media_max_size_bytes,
                    allow_private_ips=settings.allow_private_ips,
                )
            except DownloadError as e:
                _send_failure_callback(
                    request, settings, "MEDIA_DOWNLOAD_FAILED", str(e)
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Run inference through the configured analysis runner.
            try:
                output = analysis_runner.analyze_media(
                    file_bytes=file_bytes,
                    extension=ext,
                    media_type=request.media_type.value,
                )
            except Exception as e:
                _send_failure_callback(
                    request, settings, "MODEL_INFERENCE_FAILED", str(e)
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Build and send success callback
            processing_time_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = map_success_result(
                request_id=request.request_id,
                media_id=request.media_id,
                detection_result=output.detection_result,
                model_name=output.model_name,
                model_version=output.model_version,
                processing_time_ms=processing_time_ms,
                device_profile="server",
            )

            if output.harness_run_id:
                result.raw_result["harnessRunId"] = output.harness_run_id
            if output.harness_checkpoint_path:
                result.raw_result["harnessCheckpointPath"] = output.harness_checkpoint_path

            callback_url = (
                request.callback_url or settings.callback_url
            )

            try:
                send_callback(
                    result=result,
                    callback_url=callback_url,
                    service_key=settings.ai_service_key,
                    max_retries=settings.callback_max_retries,
                    base_delay=settings.callback_retry_base_delay,
                    max_delay=settings.callback_retry_max_delay,
                    timeout=settings.callback_timeout,
                    restrict_to_base_url=settings.callback_base_url if settings.restrict_callback_to_base_url else None,
                    allow_private_ips=settings.allow_private_ips,
                )
                # Acknowledge only after successful callback
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    f"Job {request.request_id} completed successfully "
                    f"in {processing_time_ms}ms",
                    extra={
                        "event": "job_completed",
                        "duration_ms": processing_time_ms,
                    },
                )
            except CallbackError:
                # Callback delivery failed after all retries — reject to DLQ
                logger.error(
                    f"Callback delivery failed for job {request.request_id}, "
                    f"rejecting to DLQ",
                    extra={"event": "job_callback_failed"},
                )
                ch.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in message: {e}",
                extra={"event": "job_invalid_json"},
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.error(
                f"Unexpected error processing message: {e}",
                exc_info=True,
                extra={"event": "job_unexpected_error"},
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        finally:
            clear_correlation_context()

    return on_message


def _send_failure_callback(request, settings, error_code, error_message):
    """Send a failure callback to the backend."""
    result = map_failure_result(
        request_id=request.request_id,
        media_id=request.media_id,
        error_code=error_code,
        error_message=error_message,
    )

    callback_url = request.callback_url or settings.callback_url

    try:
        send_callback(
            result=result,
            callback_url=callback_url,
            service_key=settings.ai_service_key,
            max_retries=settings.callback_max_retries,
            base_delay=settings.callback_retry_base_delay,
            timeout=settings.callback_timeout,
            restrict_to_base_url=settings.callback_base_url if settings.restrict_callback_to_base_url else None,
            allow_private_ips=settings.allow_private_ips,
        )
    except CallbackError:
        logger.error(
            f"Failed to deliver failure callback for {request.request_id}",
            extra={"event": "failure_callback_failed"},
        )


def start_server_consumer(analysis_runner, settings):
    """Start the server RabbitMQ consumer in a background thread.

    Args:
        analysis_runner: Initialized server analysis runner.
        settings: ServerSettings instance.
    """
    callback = create_server_consumer(analysis_runner, settings)

    def run():
        consume_with_reconnect(
            host=settings.rabbitmq_host,
            port=settings.rabbitmq_port,
            user=settings.rabbitmq_user,
            password=settings.rabbitmq_pass,
            queue_name=settings.server_queue_name,
            callback=callback,
            exchange=settings.rabbitmq_exchange,
            edge_queue=settings.edge_queue_name,
            server_queue=settings.server_queue_name,
            dlx_exchange=settings.dead_letter_exchange,
            dlq_queue=settings.dead_letter_queue,
            heartbeat=settings.rabbitmq_heartbeat,
            prefetch_count=settings.rabbitmq_prefetch_count,
        )

    thread = threading.Thread(target=run, daemon=True, name="server-consumer")
    thread.start()
    logger.info(
        "Server RabbitMQ consumer started",
        extra={"event": "consumer_started", "queue": settings.server_queue_name},
    )
    return thread
