"""One client protocol for completion calls (portolan-spec-v2.md section 11:
"every model call goes through one wrapper"). Not every model call uses it
yet: grep for `claude_agent_sdk`, `boto3` and `anthropic` imports before
assuming a call is metered here.
"""

from __future__ import annotations

from portolan.llm.agent_sdk import AgentSDKClient
from portolan.llm.base import LLMClient, LLMError, LLMJsonError, LLMResult
from portolan.llm.bedrock import BedrockClient
from portolan.llm.factory import get_llm
from portolan.llm.fake import FakeLLMClient
from portolan.llm.json_util import extract_and_validate, extract_json_object, validate_json_object

__all__ = [
    "AgentSDKClient",
    "BedrockClient",
    "FakeLLMClient",
    "LLMClient",
    "LLMError",
    "LLMJsonError",
    "LLMResult",
    "extract_and_validate",
    "extract_json_object",
    "get_llm",
    "validate_json_object",
]
