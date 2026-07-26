"""
Publish AI analysis result events to RabbitMQ.
"""

import json
from typing import Optional

import pika

from shared.messaging.topology import (
    REQUEST_BACKEND_EXCHANGE,
    RESULT_ROUTING_KEY,
)
from shared.schemas.analysis_result import AnalysisResultEvent
from shared.utils.logging import get_logger

logger = get_logger("result_publisher")


class ResultPublishError(Exception):
    """Raised when a result event cannot be published."""


def publish_analysis_result(
    channel,
    result_event: AnalysisResultEvent,
    routing_key: str = RESULT_ROUTING_KEY,
    exchange: str = REQUEST_BACKEND_EXCHANGE,
) -> None:
    """Publish a durable result event and require broker confirm."""
    payload = result_event.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=False,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    try:
        channel.confirm_delivery()
        published = channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
                message_id=result_event.event_id,
                correlation_id=result_event.correlation_id,
                timestamp=int(result_event.processed_at.timestamp()),
                type="AIAnalysisResultEvent",
            ),
            mandatory=False,
        )
    except Exception as ex:
        raise ResultPublishError(str(ex)) from ex

    if published is False:
        raise ResultPublishError("Broker did not confirm result publish")

    logger.info(
        "Published AI analysis result event",
        extra={
            "event": "analysis_result_published",
            "requestId": result_event.analysis_id,
            "status": result_event.status.value,
            "routingKey": routing_key,
        },
    )
