import asyncio
import copy
import functools
import json
import os
import random
import re
import time
import typing as T

import httpx
from anthropic import AsyncAnthropic
from devtools import debug
from google.genai import Client as GoogleGenAI
from google.genai.types import GenerateContentConfig, ThinkingConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.gateway import gateway_provider
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import assistant, image, system, user

from src.async_utils.semaphore_monitor import MonitoredSemaphore
from src.llms.models import (
    LMSTUDIO_OPENAI_BASE_URL,
    Model,
)
from src.llms.openai_responses import (
    OPENAI_MODEL_MAX_OUTPUT_TOKENS,
    create_and_poll_response,
    extract_structured_output,
)
from src.log import log
from src.run_progress import record_llm_usage
from src.utils import random_str

BMType = T.TypeVar("BMType", bound=BaseModel)


P = T.ParamSpec("P")
R = T.TypeVar("R")
COPILOT_BASE_URL = "http://localhost:4141/v1"


def retry_with_backoff(
    max_retries: int,
    base_delay: float = 3,
    max_delay: float = 120,
) -> T.Callable[[T.Callable[P, T.Awaitable[R]]], T.Callable[P, T.Awaitable[R]]]:
    """
    Decorator for *async* functions that retries transient “UNAVAILABLE /
    RESOURCE_EXHAUSTED”-style errors with exponential back-off.

    • Executes up to `max_retries + 1` total attempts (first try + N retries).
    • Full-jitter back-off — waits a random time in `[0, base_delay × 2**(n-1)]`.
    • Classifies retryability with simple string matching for readability.
    """

    def decorator(fn: T.Callable[P, T.Awaitable[R]]) -> T.Callable[P, T.Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, max_retries + 2):  # 1-based
                start = time.time()
                try:
                    res = await fn(*args, **kwargs)
                    if attempt > 1:
                        log.debug(
                            "retry_succeeded!", function=fn.__name__, attempt=attempt
                        )
                    return res
                except asyncio.CancelledError:  # never retry cancellations
                    raise

                except Exception as exc:  # noqa: BLE001
                    duration = time.time() - start
                    msg = str(exc)

                    # ---- simple, readable retry classification ----
                    retryable = (
                        "UNAVAILABLE" in msg.upper()
                        or "RESOURCE_EXHAUSTED" in msg.upper()
                        or "StatusCode.UNAVAILABLE" in msg
                        or "StatusCode.RESOURCE_EXHAUSTED" in msg
                        or "StatusCode.UNKNOWN" in msg
                        or "Empty response from OpenRouter model" in msg
                        or "validation error" in msg
                        or "SAFETY_CHECK_TYPE_BIO" in msg
                        or "524" in msg  # Cloudflare timeout
                        or "timeout" in msg.lower()
                        or "ServerError" in msg
                        or "Provider returned error" in msg
                    )
                    if "StatusCode.DEADLINE_EXCEEDED" in msg:
                        retryable = False
                    if duration > 1_000:
                        retryable = False
                    if duration > 500 and attempt > 2:
                        retryable = False

                    if not retryable or attempt > max_retries:
                        log.error(
                            "retry_failed",
                            function=fn.__name__,
                            attempt=attempt,
                            duration_seconds=duration,
                            error=msg,
                            error_type=type(exc).__name__,
                            max_retries_reached=(attempt > max_retries),
                        )
                        raise

                    # ---- full-jitter exponential back-off ----
                    retry_after_seconds: float | None = None
                    retry_after_match = re.search(
                        r"Please try again in ([0-9]*\.?[0-9]+)\s*(ms|s)\.?",
                        msg,
                        re.IGNORECASE,
                    )
                    if retry_after_match:
                        retry_after_value = float(retry_after_match.group(1))
                        retry_after_unit = retry_after_match.group(2).lower()
                        retry_after_seconds = (
                            retry_after_value / 1000
                            if retry_after_unit == "ms"
                            else retry_after_value
                        )
                        retry_after_seconds = max(retry_after_seconds, 0.05)

                    if retry_after_seconds is not None:
                        wait = retry_after_seconds
                    else:
                        base_wait = min(base_delay * 2 ** (attempt - 1), max_delay)
                        wait = random.uniform(0, base_wait)

                    log.warn(
                        "retry_attempt",
                        function=fn.__name__,
                        attempt=attempt,
                        duration_seconds=duration,
                        wait_seconds=wait,
                        error=msg,
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(wait)

            # should never reach here
            raise RuntimeError("retry_with_backoff: fell through unexpectedly")

        return wrapper

    return decorator


# openai_client = AsyncOpenAI(
#     api_key=os.environ["OPENAI_API_KEY"], timeout=10_800, max_retries=2
# )
# anthropic_client = AsyncAnthropic(
#     api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=3_010, max_retries=2
# )
# deepseek_client = AsyncOpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com",
#     timeout=2500,
#     max_retries=2,
# )
# openrouter_client = AsyncOpenAI(
#     api_key=os.environ["OPENROUTER_API_KEY"],
#     base_url="https://openrouter.ai/api/v1",
#     timeout=2500,
#     max_retries=2,
# )
groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=2500,
    max_retries=2,
)
gemini_client = GoogleGenAI(
    api_key=os.environ["GEMINI_API_KEY"],
)
kilo_client = AsyncOpenAI(
    api_key=os.environ["KILO_API_KEY"],
    base_url="https://api.kilo.ai/api/gateway",
    timeout=2500,
    max_retries=2,
)
lmstudio_client = AsyncOpenAI(
    api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
    base_url=LMSTUDIO_OPENAI_BASE_URL,
    timeout=10_800,
    max_retries=2,
)

# Semaphore to limit concurrent API calls to 100
API_SEMAPHORE = MonitoredSemaphore(
    int(os.environ["MAX_CONCURRENCY"]), name="API_SEMAPHORE"
)


async def get_next_structure(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    res_id = random_str(k=6)

    with log.span(
        "llm_call",
        model=model.value,
        structure=structure.__name__,
        request_id=res_id,
    ) as span:
        start = time.time()
        log.debug(
            "Starting LLM call",
            model=model.value,
            structure=structure.__name__,
            request_id=res_id,
        )

        async with API_SEMAPHORE:
            if model in [
                Model.o4_mini,
                Model.o3,
                Model.gpt_4_1,
                Model.gpt_4_1_mini,
                Model.o3_pro,
                Model.gpt_5,
                Model.gpt_52,
                Model.gpt_5_pro,
            ]:
                res = await _get_next_structure_openai(
                    structure=structure, model=model, messages=messages
                )
            elif model in [
                Model.copilot_gpt_5_mini,
            ]:
                res = await _get_next_structure_copilot(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.sonnet_4, Model.opus_4, Model.sonnet_4_5]:
                res = await _get_next_structure_anthropic(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.grok_4, Model.grok_3_mini_fast]:
                res = await _get_next_structure_xai(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.deepseek_reasoner, Model.deepseek_chat]:
                res = await _get_next_structure_deepseek(
                    structure=structure, model=model, messages=messages
                )
            elif model in [
                Model.gemini_2_5,
                Model.gemini_2_5_flash_lite,
                Model.gemini_3_pro,
                Model.gemini_3_flash,
            ]:
                res = await _get_next_structure_gemini(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.gemini_3_pro_gateway]:
                res = await _get_next_structure_pydantic_gateway(
                    structure=structure, model=model, messages=messages
                )
            elif model.name.startswith("kilo"):
                res = await _get_next_structure_kilo(
                    structure=structure, model=model, messages=messages
                )
            elif model.name.startswith("lmstudio"):
                res = await _get_next_structure_lmstudio(
                    structure=structure, model=model, messages=messages
                )
            elif model.name.startswith("openrouter"):
                res = await _get_next_structure_openrouter(
                    structure=structure, model=model, messages=messages
                )
            else:
                raise Exception(f"Invalid model {model}.")

            duration = time.time() - start
            span.set_attribute("duration_seconds", duration)
            # span.set_attribute("response", res.model_dump())

            response_dump = res.model_dump()
            response_keys = list(response_dump.keys())
            if os.getenv("LOG_GRIDS", "0") == "1":
                pass
            else:
                response_dump = {}

            log.debug(
                "LLM call completed",
                model=model.value,
                structure=structure.__name__,
                duration_seconds=duration,
                request_id=res_id,
                response=response_dump,
                response_keys=response_keys,
            )

            return res


async def _get_next_structure_openai(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    reasoning: dict[str, str] | None = None
    if model in [
        Model.o3,
        Model.o4_mini,
        Model.o3_pro,
        Model.gpt_5,
        Model.gpt_52,
        Model.gpt_5_pro,
    ]:
        reasoning = {"effort": "high"}

    max_output_tokens = OPENAI_MODEL_MAX_OUTPUT_TOKENS.get(model, 128_000)

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    create_kwargs: dict[str, T.Any] = {
        "model": model.value,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": structure.__name__,
                "schema": schema,
                "strict": True,
            }
        },
    }
    if reasoning:
        create_kwargs["reasoning"] = reasoning

    openai_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=10_800, max_retries=2
    )
    raw_response = await create_and_poll_response(
        openai_client,
        model=model,
        create_kwargs=create_kwargs,
    )
    openai_usage = OpenAIUsage()
    if raw_response.usage:
        u = raw_response.usage
        openai_usage = OpenAIUsage(
            output_tokens=u.output_tokens,
            input_tokens=u.input_tokens,
            total_tokens=u.total_tokens,
            reasoning_tokens=u.output_tokens_details.reasoning_tokens,
            cached_prompt_tokens=u.input_tokens_details.cached_tokens,
        )

    log.debug(
        "openai_usage",
        model=model.value,
        usage=openai_usage.model_dump(),
        cents=openai_usage.cents(model=model),
        response_status=raw_response.status,
        reasoning=raw_response.reasoning,
    )
    record_llm_usage(openai_usage.input_tokens, openai_usage.output_tokens)

    if model in [Model.o3_pro]:
        debug(raw_response.model_dump())

    payload = extract_structured_output(raw_response)
    output: BMType = structure.model_validate(payload)
    return output


def update_messages_xai(messages: list[dict]) -> list:
    final_messages = []
    for message in messages:
        if message["role"] == "system":
            role = system
        elif message["role"] == "user":
            role = user
        elif message["role"] == "assistant":
            role = assistant
        else:
            raise Exception(f"invalid role in message: {message}")
        if isinstance(message["content"], list):
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text"]:
                    final_messages.append(role(c["text"]))
                elif c["type"] == "input_image":
                    final_messages.append(role(image(c["image_url"])))
                else:
                    raise Exception(f"invalid content type: {c}")
        else:
            raise Exception(f"make sure content is a list!: {message}")
    return final_messages


def update_messages_anthropic(messages: list[dict]) -> list[dict]:
    messages = copy.deepcopy(messages)
    for message in messages:
        if "content" in message:
            if isinstance(message["content"], list):
                for c in message["content"]:
                    if c["type"] in ["input_text", "output_text"]:
                        c["type"] = "text"
                    if c["type"] == "input_image":
                        c["type"] = "image"
                        c["source"] = {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": c["image_url"].replace(
                                "data:image/png;base64,", ""
                            ),
                        }
                        del c["image_url"]
                        del c["detail"]
    return messages


MAX_TOKENS_ANTHROPIC_D: dict[Model, int] = {
    Model.sonnet_4: 64_000,
    Model.opus_4: 32_000,
    Model.sonnet_4_5: 64_000,
}
MAX_TOKENS_THINKING_ANTHROPIC_D: dict[Model, int] = {
    Model.sonnet_4: 60_000,
    Model.opus_4: 30_000,
    Model.sonnet_4_5: 60_000,
}
MAX_TOKENS_DEEPSEEK_D: dict[Model, int] = {
    Model.deepseek_chat: 8_192,
    Model.deepseek_reasoner: 32_768,
}


async def _get_next_structure_anthropic(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    anthropic_client = AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=3_010, max_retries=2
    )
    tool_schema = structure.model_json_schema()
    messages = update_messages_anthropic(messages=messages)
    response = await anthropic_client.messages.create(
        model=model.value,
        messages=messages,
        max_tokens=MAX_TOKENS_ANTHROPIC_D[model],
        tools=[
            {
                "name": "output_grid",
                "description": tool_schema["description"],
                "input_schema": tool_schema,
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": MAX_TOKENS_THINKING_ANTHROPIC_D[model],
        },
    )
    tool_call = next(block for block in response.content if block.type == "tool_use")
    tool_input = tool_call.input
    output: BMType = structure.model_validate(tool_input)
    if response.usage is not None:
        record_llm_usage(
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
    return output


class ModelPricing(BaseModel):
    prompt_tokens: float
    reasoning_tokens: float
    completion_tokens: float


MODEL_PRICING_D: dict[Model, ModelPricing] = {
    Model.grok_4: ModelPricing(
        prompt_tokens=300 / 1_000_000,
        reasoning_tokens=1_500 / 1_000_000,
        completion_tokens=1_500 / 1_000_000,
    ),
    Model.grok_3_mini_fast: ModelPricing(
        prompt_tokens=60 / 1_000_000,
        reasoning_tokens=400 / 1_000_000,
        completion_tokens=400 / 1_000_000,
    ),
    # OpenAI pricing (per million tokens)
    Model.o3: ModelPricing(
        prompt_tokens=5_000 / 1_000_000,  # $5 per 1M tokens
        reasoning_tokens=25_000 / 1_000_000,  # $25 per 1M tokens
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens
    ),
    Model.o3_pro: ModelPricing(
        prompt_tokens=1_5_00 / 1_000_000,  # $15 per 1M tokens
        reasoning_tokens=6_000 / 1_000_000,  # $60 per 1M tokens
        completion_tokens=6_000 / 1_000_000,  # $60 per 1M tokens
    ),
    Model.o4_mini: ModelPricing(
        prompt_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
        reasoning_tokens=1_200 / 1_000_000,  # $1.20 per 1M tokens
        completion_tokens=1_200 / 1_000_000,  # $1.20 per 1M tokens
    ),
    Model.gpt_4_1: ModelPricing(
        prompt_tokens=250 / 1_000_000,  # $2.50 per 1M tokens
        reasoning_tokens=1_000 / 1_000_000,  # $10 per 1M tokens
        completion_tokens=1_000 / 1_000_000,  # $10 per 1M tokens
    ),
    Model.gpt_4_1_mini: ModelPricing(
        prompt_tokens=150 / 1_000_000,  # $0.15 per 1M tokens
        reasoning_tokens=600 / 1_000_000,  # $0.60 per 1M tokens
        completion_tokens=600 / 1_000_000,  # $0.60 per 1M tokens
    ),
    Model.gpt_5: ModelPricing(
        prompt_tokens=125 / 1_000_000,  # $10 per 1M tokens (estimate)
        reasoning_tokens=1_000 / 1_000_000,  # $50 per 1M tokens (estimate)
        completion_tokens=1_000 / 1_000_000,  # $30 per 1M tokens (estimate)
    ),
    Model.gpt_52: ModelPricing(
        prompt_tokens=175 / 1_000_000,  # $1.75 per 1M tokens
        reasoning_tokens=1_400 / 1_000_000,  # $14 per 1M tokens (same as output)
        completion_tokens=1_400 / 1_000_000,  # $14 per 1M tokens
    ),
    Model.gpt_5_pro: ModelPricing(
        prompt_tokens=200 / 1_000_000,  # $20 per 1M tokens (estimate)
        reasoning_tokens=1_200 / 1_000_000,  # $60 per 1M tokens (estimate)
        completion_tokens=1_200 / 1_000_000,  # $60 per 1M tokens (estimate)
    ),
    Model.sonnet_4_5: ModelPricing(
        prompt_tokens=3_000 / 1_000_000,  # $10 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $50 per 1M tokens (estimate)
        completion_tokens=15_000 / 1_000_000,  # $30 per 1M tokens (estimate)
    ),
    # Gemini pricing (per million tokens)
    Model.gemini_2_5: ModelPricing(
        prompt_tokens=1_250 / 1_000_000,  # $1.25 per 1M tokens
        reasoning_tokens=10_000 / 1_000_000,  # $10 per 1M tokens (thinking)
        completion_tokens=10_000 / 1_000_000,  # $10 per 1M tokens
    ),
    Model.gemini_2_5_flash_lite: ModelPricing(
        prompt_tokens=75 / 1_000_000,  # $0.075 per 1M tokens
        reasoning_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
        completion_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
    ),
    Model.gemini_3_flash: ModelPricing(
        prompt_tokens=2_500 / 1_000_000,  # $2.50 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (thinking, estimate)
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (estimate)
    ),
    Model.gemini_3_pro: ModelPricing(
        prompt_tokens=2_500 / 1_000_000,  # $2.50 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (thinking, estimate)
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (estimate)
    ),
    # Pydantic AI Gateway - same pricing as direct, gateway is free during beta
    Model.gemini_3_pro_gateway: ModelPricing(
        prompt_tokens=2_500 / 1_000_000,  # $2.50 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (thinking, estimate)
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (estimate)
    ),
    # OpenRouter Gemini 3 Pro - $2/M input, $12/M output (from OpenRouter)
    Model.gemini_3_pro_openrouter: ModelPricing(
        prompt_tokens=2_000 / 1_000_000,  # $2 per 1M tokens
        reasoning_tokens=0
        / 1_000_000,  # OpenRouter doesn't charge separately for reasoning
        completion_tokens=12_000 / 1_000_000,  # $12 per 1M tokens
    ),
    Model.lmstudio_qwen_3_5_27b: ModelPricing(
        prompt_tokens=0.0,
        reasoning_tokens=0.0,
        completion_tokens=0.0,
    ),
}


class GrokUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_text_tokens: int
    reasoning_tokens: int
    cached_prompt_text_tokens: int

    def cents(self, model: Model) -> int:
        pricing = MODEL_PRICING_D[model]
        return round(
            self.prompt_tokens * pricing.prompt_tokens
            + self.reasoning_tokens * pricing.reasoning_tokens
            + self.completion_tokens * pricing.completion_tokens
        )


class OpenAIUsage(BaseModel):
    output_tokens: int = 0
    input_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0

    def cents(self, model: Model) -> float:
        if model not in MODEL_PRICING_D:
            return 0.0
        pricing = MODEL_PRICING_D[model]
        return round(
            self.input_tokens * pricing.prompt_tokens
            + self.reasoning_tokens * pricing.reasoning_tokens
            + self.output_tokens * pricing.completion_tokens,
            2,
        )


def _optional_int(n: int | None) -> int:
    return 0 if n is None else n


def _openai_usage_from_completion_usage(usage: CompletionUsage | None) -> OpenAIUsage:
    if usage is None:
        return OpenAIUsage()
    comp = usage.completion_tokens_details
    prompt = usage.prompt_tokens_details
    return OpenAIUsage(
        output_tokens=usage.completion_tokens,
        input_tokens=usage.prompt_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=_optional_int(comp.reasoning_tokens) if comp else 0,
        cached_prompt_tokens=_optional_int(prompt.cached_tokens) if prompt else 0,
    )


def _chat_message_reasoning_content(msg: ChatCompletionMessage) -> str | None:
    """LM Studio and some servers add `reasoning_content`; it is not on the base OpenAI type."""
    extra = getattr(msg, "reasoning_content", None)
    return extra if isinstance(extra, str) else None


class GeminiUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int = 0
    cached_tokens: int = 0

    def cents(self, model: Model) -> float:
        if model not in MODEL_PRICING_D:
            return 0.0
        pricing = MODEL_PRICING_D[model]
        return round(
            self.prompt_tokens * pricing.prompt_tokens
            + self.thinking_tokens * pricing.reasoning_tokens
            + self.completion_tokens * pricing.completion_tokens,
            2,
        )


@retry_with_backoff(max_retries=20)
async def _get_next_structure_xai(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    messages = update_messages_xai(messages=messages)

    api_keys = os.environ["XAI_API_KEY"].split(",")
    xai_client = XaiAsyncClient(
        api_key=random.choice(api_keys),
        timeout=3_010,
        channel_options=[
            # ("grpc.service_config", custom_retry_policy),
        ],
    )
    chat = xai_client.chat.create(
        model=model.value,
        messages=messages,
        max_tokens=256_000,
        # reasoning_effort="high", # not supported for grok-4
    )
    response, struct = await chat.parse(shape=structure)
    try:
        grok_usage = GrokUsage(
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            prompt_text_tokens=response.usage.prompt_text_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cached_prompt_text_tokens=response.usage.cached_prompt_text_tokens,
        )
        log.debug(
            "usage",
            usage=grok_usage,
            cents=grok_usage.cents(model=model),
            finish_reason=response.finish_reason,
            reasoning_content=response.reasoning_content
            if os.getenv("LOG_GRIDS", "0") == "1"
            else None,
        )
        record_llm_usage(
            grok_usage.prompt_text_tokens,
            grok_usage.completion_tokens,
        )

    except Exception as e:
        print(f"usage error: {e=}")
        pass
    return struct


def update_messages_deepseek(
    messages: list[dict], structure: type[BMType]
) -> list[dict]:
    messages = copy.deepcopy(messages)
    schema = structure.model_json_schema()

    # Convert messages to simple format expected by DeepSeek
    final_messages = []

    # Add system message with JSON instructions
    system_content = f"""You are a helpful assistant that outputs structured JSON data.
Always output valid JSON that strictly follows this schema:
{schema}

IMPORTANT: Give the output in a valid JSON string (it should not be wrapped in markdown, just plain json object)."""

    final_messages.append({"role": "system", "content": system_content})

    for message in messages:
        if message["role"] == "system":
            # Append to our system message
            final_messages[0]["content"] += f"\n\n{message.get('content', '')}"
        else:
            if isinstance(message["content"], list):
                # Concatenate all text content
                text_parts = []
                for c in message["content"]:
                    if c["type"] in ["input_text", "output_text"]:
                        text_parts.append(c["text"])
                content = " ".join(text_parts)
            else:
                content = message["content"]

            final_messages.append({"role": message["role"], "content": content})

    return final_messages


async def _get_next_structure_deepseek(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    deepseek_client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=2500,
        max_retries=2,
    )
    messages = update_messages_deepseek(messages=messages, structure=structure)

    # Use JSON mode
    response = await deepseek_client.chat.completions.create(
        model=model.value,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS_DEEPSEEK_D[model],
        # temperature=0.3,  # Lower temperature for more consistent JSON output
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from DeepSeek model")

    if response.usage is not None:
        record_llm_usage(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


@retry_with_backoff(max_retries=20)
async def _get_next_structure_copilot(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    copilot_client = AsyncOpenAI(
        api_key=os.environ.get("COPILOT_API_KEY") or "copilot",
        base_url=COPILOT_BASE_URL,
        timeout=2500,
        max_retries=2,
    )
    messages = update_messages_openrouter(messages=messages)

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {
            "name": structure.__name__,
            "strict": True,
            "schema": schema,
        },
    }

    content: str | None = ""
    try:
        response = await copilot_client.chat.completions.create(
            model=model.value, messages=messages, response_format=response_format
        )
        _cu = _openai_usage_from_completion_usage(response.usage)
        record_llm_usage(_cu.input_tokens, _cu.output_tokens)

        if not response.choices:
            raise Exception(f"Copilot returned no choices: {response}")
        content = response.choices[0].message.content
        if not content:
            raise Exception("Empty response from Copilot model")

        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except Exception as e:
        raise Exception(
            f"Failed to parse structured Copilot response for {model.value}. Error: {e} content: {content}"
        )


def update_messages_openrouter(
    messages: list[dict],
    structure: type[BMType] | None = None,
    use_json_object: bool = False,
) -> list[dict]:
    """Convert messages to OpenRouter format, optionally with schema instructions for json_object mode."""
    messages = copy.deepcopy(messages)
    final_messages = []

    # If using json_object mode (not json_schema), we need to add instructions
    if use_json_object and structure:
        schema = structure.model_json_schema()
        system_content = f"""You are a helpful assistant that outputs structured JSON data.
Always output valid JSON that strictly follows this schema:
{schema}

IMPORTANT: Give the output in a valid JSON string (it should not be wrapped in markdown, just plain json object)."""
        final_messages.append({"role": "system", "content": system_content})

    for message in messages:
        if isinstance(message["content"], list):
            # Handle structured content format
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
            content = " ".join(text_parts)
        else:
            content = message["content"]

        # If we added a system message for json_object mode, append to it
        if use_json_object and structure and message["role"] == "system":
            final_messages[0]["content"] += f"\n\n{content}"
        else:
            final_messages.append({"role": message["role"], "content": content})

    return final_messages


def update_messages_gemini(messages: list[dict]) -> str:
    """Convert messages to a single prompt string for Gemini."""
    parts = []

    for message in messages:
        role = message["role"]

        if isinstance(message["content"], list):
            # Handle structured content format
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
                # Note: For now, we're skipping image handling for Gemini
                # You can add image support later if needed
            content = " ".join(text_parts)
        else:
            content = message["content"]

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_openrouter(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    openrouter_client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=2500,
        max_retries=2,
    )

    # Check if we need to use json_object mode
    use_json_object = False
    if model in [
        Model.openrouter_qwen_235b_thinking,
        Model.openrouter_glm,
    ]:
        use_json_object = True

    messages = update_messages_openrouter(
        messages=messages,
        structure=structure if use_json_object else None,
        use_json_object=use_json_object,
    )

    # Get the JSON schema for the structure
    schema = structure.model_json_schema()

    # Ensure additionalProperties is set to false for strict validation
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    # Set response format based on mode
    response_format: ResponseFormat
    if use_json_object:
        response_format = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": structure.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    extra_body = {}

    # Special cases for certain models
    if model in [Model.openrouter_qwen_235b_thinking]:
        # This model might not support structured outputs
        extra_body["provider"] = {
            "order": ["Novita"],
            "allow_fallbacks": True,
        }
    elif model in [
        Model.openrouter_qwen_235b,
        # Model.openrouter_qwen_235b_thinking,
    ]:
        extra_body["provider"] = {
            "only": ["cerebras"],
            # "allow_fallbacks": True,
        }
    elif model == Model.gemini_3_pro_openrouter:
        # Gemini 3 Pro via OpenRouter - force Google provider, enable reasoning
        extra_body["provider"] = {
            "only": ["Google"],
            "allow_fallbacks": False,
        }
        # OpenRouter's reasoning parameter for thinking models
        extra_body["reasoning"] = {"effort": "high"}

    # if model in [Model.openrouter_glm]:
    #     extra_body["reasoning"]["enabled"] = True

    response = await openrouter_client.chat.completions.create(
        model=model.value,
        messages=messages,
        response_format=response_format,
        max_tokens=100_000,  # Default max tokens for OpenRouter
        # temperature=0.3,  # Lower temperature for more consistent JSON output
        extra_body=extra_body,
        # reasoning_effort="high",
    )

    _ou = _openai_usage_from_completion_usage(response.usage)
    record_llm_usage(_ou.input_tokens, _ou.output_tokens)

    # Parse the JSON response
    if not response.choices:
        raise Exception(f"OpenRouter returned no choices: {response}")
    content = response.choices[0].message.content
    if not content:
        # debug(response)
        raise Exception("Empty response from OpenRouter model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


def _log_lmstudio_completion_parse_failure(
    *,
    reason: str,
    response: ChatCompletion,
    api_model: str,
    structure_name: str,
) -> None:
    """Log fields that explain missing or unusable assistant text (LM Studio / OpenAI-compatible)."""
    details: dict[str, T.Any] = {
        "reason": reason,
        "api_model": api_model,
        "structure": structure_name,
        "response_id": response.id,
        "response_model_field": response.model,
    }
    if response.usage is not None:
        details["usage"] = response.usage.model_dump()

    choices = response.choices
    details["choices_len"] = len(choices)
    if not choices:
        log.error("LM Studio structured call: no choices", **details)
        return

    ch0 = choices[0]
    details["finish_reason"] = ch0.finish_reason
    details["choice_index"] = ch0.index
    msg = ch0.message

    try:
        assistant = msg.model_dump()
        raw_content = assistant.get("content")
        if isinstance(raw_content, str):
            assistant["content_len"] = len(raw_content)
            if raw_content:
                assistant["content_head"] = raw_content[:240]
        details["assistant_message"] = assistant
    except Exception as e:
        details["assistant_message_dump_error"] = repr(e)

    log.error("LM Studio structured call: unusable assistant content", **details)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_lmstudio(
    structure: type[BMType],
    model: Model,
    messages: list,
) -> BMType:
    messages = update_messages_openrouter(messages=messages)
    api_model = model.value

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {
            "name": structure.__name__,
            "strict": True,
            "schema": schema,
        },
    }

    response = await lmstudio_client.chat.completions.create(
        model=api_model,
        messages=messages,
        response_format=response_format,
    )

    openai_usage = _openai_usage_from_completion_usage(response.usage)
    record_llm_usage(openai_usage.input_tokens, openai_usage.output_tokens)
    if openai_usage is not None and response.choices:
        ch0 = response.choices[0]
        log.info(
            "lmstudio_usage",
            model=api_model,
            usage=openai_usage.model_dump(),
            cents=openai_usage.cents(model=model),
            finish_reason=ch0.finish_reason,
            reasoning_content=_chat_message_reasoning_content(ch0.message),
        )

    if not response.choices:
        _log_lmstudio_completion_parse_failure(
            reason="no_choices",
            response=response,
            api_model=api_model,
            structure_name=structure.__name__,
        )
        raise Exception(f"LM Studio returned no choices: {response}")
    msg = response.choices[0].message
    raw_content = msg.content
    if isinstance(raw_content, str):
        content = raw_content.strip()
    elif raw_content is None:
        content = ""
    else:
        content = str(raw_content).strip()

    if not content:
        reasoning = _chat_message_reasoning_content(msg)
        if reasoning and reasoning.strip():
            content = reasoning.strip()
            log.debug(
                "LM Studio: using reasoning_content for structured JSON (message.content empty)",
                api_model=api_model,
                structure=structure.__name__,
            )

    if not content:
        _log_lmstudio_completion_parse_failure(
            reason="empty_or_missing_message_content",
            response=response,
            api_model=api_model,
            structure_name=structure.__name__,
        )
        raise Exception("Empty response from LM Studio model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    m = re.match(
        r"^```(?:json)?\s*\r?\n?(.*)\r?\n?```\s*$", t, re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return t


def _extract_balanced_json_fragment(
    text: str, start: int, open_ch: str, close_ch: str
) -> str | None:
    if start >= len(text) or text[start] != open_ch:
        return None
    depth = 1
    in_string = False
    escape = False
    i = start + 1
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _loads_json_from_model_text(content: str) -> T.Any:
    """Parse JSON from chat completions; tolerate markdown fences and trailing prose."""
    raw = content.strip().replace("\ufeff", "")
    raw = _strip_markdown_json_fence(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        idx = 0
        while True:
            pos = raw.find(open_ch, idx)
            if pos < 0:
                break
            frag = _extract_balanced_json_fragment(raw, pos, open_ch, close_ch)
            if frag:
                try:
                    return json.loads(frag)
                except json.JSONDecodeError:
                    pass
            idx = pos + 1
    raise json.JSONDecodeError(
        "no parseable JSON object or array in model content", content, 0
    )


@retry_with_backoff(max_retries=20)
async def _get_next_structure_kilo(
    structure: type[BMType],
    model: Model,
    messages: list,
) -> BMType:
    messages = update_messages_openrouter(messages=messages)
    _json_hint = "Respond with JSON matching the requested schema."
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{_json_hint}\n\n{messages[0]['content']}"
    else:
        messages.insert(0, {"role": "system", "content": _json_hint})

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {
            "name": structure.__name__,
            "strict": True,
            "schema": schema,
        },
    }

    response = await kilo_client.chat.completions.create(
        model=model.value,
        messages=messages,
        response_format=response_format,
        max_tokens=100_000,
    )
    _ku = _openai_usage_from_completion_usage(response.usage)
    record_llm_usage(_ku.input_tokens, _ku.output_tokens)

    if not response.choices:
        raise Exception(f"Kilo returned no choices: {response}")
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from Kilo model")

    try:
        json_data = _loads_json_from_model_text(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


# Gemini model output token limits
GEMINI_MODEL_MAX_OUTPUT_TOKENS: dict[Model, int] = {
    Model.gemini_2_5: 65_536,
    Model.gemini_2_5_flash_lite: 8_192,
    Model.gemini_3_pro: 65_536,  # Max output tokens
}


@retry_with_backoff(max_retries=20)
async def _get_next_structure_gemini(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    # Convert messages to Gemini format
    prompt = update_messages_gemini(messages=messages)

    # Build config for structured output with maxed out settings
    config = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=structure,
        max_output_tokens=GEMINI_MODEL_MAX_OUTPUT_TOKENS.get(model, 65_536),
    )

    # Enable thinking for reasoning models (Gemini 3 Pro)
    config.thinking_config = ThinkingConfig(thinking_level="HIGH")

    # Use native async API instead of asyncio.to_thread
    response = await gemini_client.aio.models.generate_content(
        model=model.value,
        contents=prompt,
        config=config,
    )

    # Extract and log usage metadata
    usage_metadata = response.usage_metadata
    if usage_metadata:
        gemini_usage = GeminiUsage(
            prompt_tokens=usage_metadata.prompt_token_count or 0,
            completion_tokens=usage_metadata.candidates_token_count or 0,
            total_tokens=usage_metadata.total_token_count or 0,
            thinking_tokens=usage_metadata.thoughts_token_count or 0,
            cached_tokens=usage_metadata.cached_content_token_count or 0,
        )
        log.debug(
            "gemini_usage",
            model=model.value,
            usage=gemini_usage.model_dump(),
            cents=gemini_usage.cents(model=model),
        )
        record_llm_usage(
            gemini_usage.prompt_tokens,
            gemini_usage.completion_tokens + gemini_usage.thinking_tokens,
        )

    # The response.parsed should contain the instantiated object
    if hasattr(response, "parsed") and response.parsed:
        return T.cast(BMType, response.parsed)

    # Fallback to parsing the text response
    content = response.text
    if not content:
        raise Exception("Empty response from Gemini model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


def update_messages_pydantic_ai(messages: list[dict]) -> str:
    """Convert messages to a single prompt string for Pydantic AI Agent.run()."""
    parts = []

    for message in messages:
        role = message["role"]

        if isinstance(message["content"], list):
            # Handle structured content format (OpenAI-style)
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
            content = " ".join(text_parts)
        else:
            content = message["content"]

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


# Pydantic AI Gateway settings for Gemini models (Vertex AI limits)
# Vertex AI has different limits than direct Gemini API: thinking_budget 1-32768
PYDANTIC_GATEWAY_THINKING_BUDGET: dict[Model, int] = {
    Model.gemini_3_pro_gateway: 32_768,  # Max for Vertex AI
}

PYDANTIC_GATEWAY_MAX_TOKENS: dict[Model, int] = {
    Model.gemini_3_pro_gateway: 65_536,  # Max output tokens
}


@retry_with_backoff(max_retries=20)
async def _get_next_structure_pydantic_gateway(
    structure: type[BMType],
    model: Model,
    messages: list,
) -> BMType:
    """
    Use Pydantic AI Gateway to call Gemini models with native thinking_config support.
    Requires PYDANTIC_AI_GATEWAY_API_KEY environment variable.
    """
    # Build model settings with thinking config
    thinking_budget = PYDANTIC_GATEWAY_THINKING_BUDGET.get(model, 65_535)
    max_tokens = PYDANTIC_GATEWAY_MAX_TOKENS.get(model, 65_536)

    settings = GoogleModelSettings(
        max_tokens=max_tokens,
        google_thinking_config={"thinking_budget": thinking_budget},
    )

    # Create gateway provider with explicit API key
    # Support both env var names
    gateway_api_key = os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY") or os.environ.get(
        "PYDANTIC_API_GATEWAY_API_KEY"
    )
    if not gateway_api_key:
        raise ValueError(
            "Set PYDANTIC_AI_GATEWAY_API_KEY or PYDANTIC_API_GATEWAY_API_KEY environment variable"
        )

    # Create custom HTTP client with long timeout for reasoning models (3 hours like GPT-5-Pro)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(10_800.0))

    # gateway_provider takes upstream provider as string (e.g., "google-vertex")
    gateway = gateway_provider(
        "google-vertex", api_key=gateway_api_key, http_client=http_client
    )

    # Extract model name from the gateway path (e.g., "gateway/google-vertex:gemini-3-pro-preview" -> "gemini-3-pro-preview")
    model_name = model.value.split(":")[-1]
    google_model = GoogleModel(model_name, provider=gateway)

    # Create agent with structured output type
    agent: Agent[None, BMType] = Agent(
        google_model,
        output_type=structure,
        model_settings=settings,
    )

    # Convert messages to prompt string
    prompt = update_messages_pydantic_ai(messages)

    # Run the agent
    result = await agent.run(prompt)

    # Log usage if available
    usage = result.usage()
    if usage:
        gemini_usage = GeminiUsage(
            prompt_tokens=usage.request_tokens or 0,
            completion_tokens=usage.response_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            thinking_tokens=0,  # Pydantic AI doesn't expose thinking tokens separately yet
            cached_tokens=0,
        )
        log.debug(
            "pydantic_gateway_usage",
            model=model.value,
            usage=gemini_usage.model_dump(),
            cents=gemini_usage.cents(model=model),
        )
        record_llm_usage(gemini_usage.prompt_tokens, gemini_usage.completion_tokens)

    return result.output


async def main_test() -> None:
    class Reasoning(BaseModel):
        """Reasoning over a problem returning the reasoning string and the answer."""

        reasoning: str = Field(..., description="Reasoning for the math problem.")
        answer: int = Field(..., description="The answer to the math problem.")

    # test openai
    """
    response = await _get_next_structure_openai(
        structure=Reasoning,
        model=Model.o4_mini,
        messages=[
            {
                "role": "system",
                "content": "you are a math solving pirate. always talk like a pirate.",
            },
            {"role": "user", "content": "what is 39 * 28937?"},
        ],
    )
    debug(response)

    # test anthropic
    response = await _get_next_structure_anthropic(
        structure=Reasoning,
        model=Model.sonnet_4,
        messages=[
            {
                "role": "user",
                "content": "you are a math solving pirate. always talk like a pirate.",
            },
            {"role": "user", "content": "what is 39 * 28937?"},
        ],
    )
    debug(response)
    """
    # test groq with structured outputs
    response = await _get_next_structure_openrouter(
        structure=Reasoning,
        model=Model.openrouter_gpt_oss_120b,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "you are a math solving pirate. always talk like a pirate.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "what is 39 * 28937?"}],
            },
        ],
    )
    debug(response)

    # test gemini
    # response = await get_next_structure(
    #     structure=Reasoning,
    #     model=Model.gemini_2_5_flash_lite,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "you are a math solving pirate. always talk like a pirate.",
    #         },
    #         {"role": "user", "content": "what is 39 * 28937?"},
    #     ],
    # )
    # debug(response)


if __name__ == "__main__":
    asyncio.run(main_test())
