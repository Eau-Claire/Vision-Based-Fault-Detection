from edge.context.models import ExecutionContext
from edge.prompts.schemas import PromptMessage, PromptMessageBundle
from edge.prompts.templates import PROMPT_VERSION, SYSTEM_TEMPLATE


class PromptBuilder:
    def build_action_prompt(self, context: ExecutionContext) -> PromptMessageBundle:
        return PromptMessageBundle(
            version=PROMPT_VERSION,
            messages=[
                PromptMessage(role="system", content=SYSTEM_TEMPLATE),
                PromptMessage(role="user", content=f"Goal: {context.goal}"),
            ],
        )
