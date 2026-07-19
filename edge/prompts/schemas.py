from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptMessage:
    role: str
    content: str


@dataclass(frozen=True)
class PromptMessageBundle:
    version: str
    messages: List[PromptMessage]
