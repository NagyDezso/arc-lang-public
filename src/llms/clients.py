import os
import random

import httpx
from anthropic import AsyncAnthropic
from google.genai import Client
from openai import AsyncOpenAI
from pydantic_ai.providers.gateway import gateway_provider
from xai_sdk import AsyncClient

COPILOT_BASE_URL = "http://localhost:4141/v1"
LMSTUDIO_BASE_URL = "http://127.0.0.1:4444/v1"

openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "openai"), timeout=10_800, max_retries=2
)
anthropic_client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "anthropic"),
    timeout=3_010,
    max_retries=2,
)
deepseek_client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "deepseek"),
    base_url="https://api.deepseek.com",
    timeout=2500,
    max_retries=2,
)
openrouter_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", "openrouter"),
    base_url="https://openrouter.ai/api/v1",
    timeout=2500,
    max_retries=2,
)
groq_client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "groq"),
    base_url="https://api.groq.com/openai/v1",
    timeout=2500,
    max_retries=2,
)
gemini_client = Client(
    api_key=os.environ.get("GEMINI_API_KEY", "gemini"),
)
kilo_client = AsyncOpenAI(
    api_key=os.environ.get("KILO_API_KEY", "kilo"),
    base_url="https://api.kilo.ai/api/gateway",
    timeout=2500,
    max_retries=2,
)
lmstudio_client = AsyncOpenAI(
    api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
    base_url=LMSTUDIO_BASE_URL,
    timeout=10_800,
    max_retries=2,
)
copilot_client = AsyncOpenAI(
    api_key=os.environ.get("COPILOT_API_KEY") or "copilot",
    base_url=COPILOT_BASE_URL,
    timeout=2500,
    max_retries=2,
)
api_keys = os.environ.get("XAI_API_KEY", "xai").split(",")
xai_client = AsyncClient(
    api_key=random.choice(api_keys),
    timeout=3_010,
    channel_options=[
        # ("grpc.service_config", custom_retry_policy),
    ],
)

# Create gateway provider with explicit API key
# Support both env var names
gateway_api_key = os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY") or os.environ.get(
    "PYDANTIC_API_GATEWAY_API_KEY", "pydantic"
)
# Create custom HTTP client with long timeout for reasoning models (3 hours like GPT-5-Pro)
http_client = httpx.AsyncClient(timeout=httpx.Timeout(10_800.0))
# gateway_provider takes upstream provider as string (e.g., "google-vertex")
gateway_client = None
if gateway_api_key:
    gateway_client = gateway_provider(
        "google-vertex", api_key=gateway_api_key, http_client=http_client
    )
