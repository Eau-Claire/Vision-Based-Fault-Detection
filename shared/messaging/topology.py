"""
RabbitMQ topology constants for PMS AI request/result flows.
"""

REQUEST_BACKEND_EXCHANGE = "identity-exchange"

REQUEST_SERVER_ROUTING_KEY = "identity.event.aianalysisrequestedevent.server"
REQUEST_EDGE_ROUTING_KEY = "identity.event.aianalysisrequestedevent.edge"

RESULT_ROUTING_KEY = "identity.event.aianalysisresultevent"
RESULT_QUEUE = "ai.analysis.result"
RESULT_RETRY_QUEUE = "ai.analysis.result.retry"
RESULT_DLQ = "ai.analysis.result.dead-letter"

