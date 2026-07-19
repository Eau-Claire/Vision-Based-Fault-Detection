"""
Callback service — sends analysis results to the ASP.NET Core backend.

Implements exponential backoff retry for reliability.
"""

import time
import json
from typing import Optional

import requests

from urllib.parse import urlparse
from shared.schemas.analysis_result import AnalysisResult
from shared.utils.logging import get_logger
from shared.utils.security import is_safe_url

logger = get_logger("callback_service")


class CallbackError(Exception):
    """Raised when callback delivery fails after all retries."""
    pass


def send_callback(
    result: AnalysisResult,
    callback_url: str,
    service_key: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    timeout: int = 15,
    restrict_to_base_url: Optional[str] = None,
    allow_private_ips: bool = True,
) -> bool:
    """Send analysis result to the backend callback API with retry.

    Args:
        result: The AnalysisResult payload to send.
        callback_url: Full URL of the callback endpoint.
        service_key: Service-to-service authentication key.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries (seconds).
        max_delay: Maximum delay cap for exponential backoff.
        timeout: HTTP request timeout in seconds.
        restrict_to_base_url: Base URL to restrict callback destination domain.
        allow_private_ips: Whether private/loopback IP addresses are permitted.

    Returns:
        True if callback was delivered successfully.

    Raises:
        CallbackError: If all retry attempts are exhausted.
    """
    # Restrict to configured base URL domain/host to prevent key leakage (X-AI-Service-Key)
    allowed_hosts = None
    if restrict_to_base_url:
        parsed_base = urlparse(restrict_to_base_url)
        if parsed_base.hostname:
            allowed_hosts = [parsed_base.hostname]

    if not is_safe_url(callback_url, allowed_hosts=allowed_hosts, allow_private_ips=allow_private_ips):
        raise CallbackError(f"Callback URL is unsafe or not allowed: {callback_url}")

    headers = {
        "X-AI-Service-Key": service_key,
        "Content-Type": "application/json",
    }

    payload = result.model_dump(by_alias=True, exclude_none=True)
    payload_json = json.dumps(payload, ensure_ascii=False)
    detection_count = len(result.detections or [])
    category_codes = sorted({det.category_code for det in result.detections or []})
    logger.info(
        "Prepared callback payload: "
        f"status={result.status.value}, detections={detection_count}",
        extra={
            "event": "callback_payload_prepared",
            "analysis_status": result.status.value,
            "detection_count": detection_count,
            "category_codes": category_codes,
            "model_name": result.model_name,
            "error_code": result.error_code,
        },
    )

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Retry attempt {attempt}/{max_retries} "
                    f"after {delay:.1f}s delay",
                    extra={
                        "event": "callback_retry",
                        "status": f"attempt_{attempt}",
                    },
                )
                time.sleep(delay)

            logger.info(
                f"Sending callback to {callback_url} "
                f"(attempt {attempt + 1}/{max_retries + 1})",
                extra={"event": "callback_send"},
            )

            response = requests.post(
                callback_url,
                data=payload_json,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code in (200, 201, 202, 204):
                logger.info(
                    f"Callback delivered successfully: "
                    f"status={response.status_code}",
                    extra={
                        "event": "callback_success",
                        "status": str(response.status_code),
                    },
                )
                return True

            # Non-retryable client errors (4xx except 408, 429)
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                logger.error(
                    f"Callback rejected (non-retryable): "
                    f"status={response.status_code}, body={response.text[:500]}",
                    extra={
                        "event": "callback_rejected",
                        "status": str(response.status_code),
                    },
                )
                raise CallbackError(
                    f"Callback rejected with status {response.status_code}: "
                    f"{response.text[:200]}"
                )

            # Server errors (5xx) or retryable client errors — retry
            last_error = CallbackError(
                f"Callback failed with status {response.status_code}"
            )
            logger.warning(
                f"Callback failed: status={response.status_code}, "
                f"will retry",
                extra={"event": "callback_failed"},
            )

        except requests.exceptions.Timeout:
            last_error = CallbackError(
                f"Callback timed out after {timeout}s"
            )
            logger.warning(
                "Callback request timed out",
                extra={"event": "callback_timeout"},
            )

        except requests.exceptions.ConnectionError as e:
            last_error = CallbackError(f"Connection error: {e}")
            logger.warning(
                f"Callback connection error: {e}",
                extra={"event": "callback_connection_error"},
            )

        except CallbackError:
            raise

        except Exception as e:
            last_error = CallbackError(f"Unexpected error: {e}")
            logger.error(
                f"Unexpected callback error: {e}",
                exc_info=True,
                extra={"event": "callback_unexpected_error"},
            )

    # All retries exhausted
    error_msg = (
        f"Callback delivery failed after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
    logger.error(error_msg, extra={"event": "callback_exhausted"})
    raise CallbackError(error_msg)
