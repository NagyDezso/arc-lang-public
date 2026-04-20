from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import random
import re
import time
import typing as T

from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    ThinkingConfigDict,
    ThinkingLevel,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from xai_sdk.chat import assistant, image, system, user

from src.async_utils.semaphore_monitor import MonitoredSemaphore
from src.llms.clients import (
    anthropic_client,
    copilot_client,
    deepseek_client,
    gateway_client,
    gemini_client,
    groq_client,
    kilo_client,
    lmstudio_client,
    openai_client,
    openrouter_client,
    xai_client,
)
from src.llms.models import (
    TokenUsage,
    parse_llm,
)
from src.llms.openai_responses import (
    create_and_poll_response,
    extract_structured_output,
)
from src.log import log
from src.logging_config import record_llm_token_usage
from src.utils import random_str

BMType = T.TypeVar("BMType", bound=BaseModel)


P = T.ParamSpec("P")
R = T.TypeVar("R")


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
                        or "Context size has been exceeded." in msg
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


API_SEMAPHORE = MonitoredSemaphore(
    int(os.environ["MAX_CONCURRENCY"]), name="API_SEMAPHORE"
)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_openai(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    reasoning: dict[str, str] | None = None
    if model_id in {
        "o3",
        "o4-mini",
        "o3-pro",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-pro",
    }:
        reasoning = {"effort": "high"}

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    create_kwargs: dict[str, T.Any] = {
        "model": model_id,
        "input": messages,
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

    raw_response = await create_and_poll_response(
        openai_client,
        model_id=model_id,
        create_kwargs=create_kwargs,
    )
    usage = TokenUsage.from_responses_api(raw_response.usage)

    payload = extract_structured_output(raw_response)
    output = structure.model_validate(payload)
    return output, usage


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


@retry_with_backoff(max_retries=20)
async def _get_next_structure_anthropic(
    structure: type[BMType],  # type[T]
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    tool_schema = structure.model_json_schema()
    messages = update_messages_anthropic(messages=messages)
    response = await anthropic_client.messages.create(
        max_tokens=128_000,
        model=model_id,
        messages=messages,
        tools=[
            {
                "name": "output_grid",
                "description": tool_schema["description"],
                "input_schema": tool_schema,
            }
        ],
    )
    tool_call = next(block for block in response.content if block.type == "tool_use")
    tool_input = tool_call.input
    output = structure.model_validate(tool_input)
    usage = TokenUsage.from_anthropic(response.usage)
    return output, usage


def _chat_message_reasoning_content(msg: ChatCompletionMessage) -> str | None:
    """LM Studio and some servers add `reasoning_content`; it is not on the base OpenAI type."""
    extra = getattr(msg, "reasoning_content", None)
    return extra if isinstance(extra, str) else None


@retry_with_backoff(max_retries=20)
async def _get_next_structure_xai(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    messages = update_messages_xai(messages=messages)

    chat = xai_client.chat.create(
        model=model_id,
        messages=messages,
    )
    response, struct = await chat.parse(shape=structure)
    token_usage = TokenUsage.from_xai_grok(response.usage)
    return struct, token_usage


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
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    messages = update_messages_deepseek(messages=messages, structure=structure)

    # Use JSON mode
    response = await deepseek_client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format={"type": "json_object"},
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from DeepSeek model")

    output = structure.model_validate_json(content)
    usage = TokenUsage.from_chat_completion(response.usage)
    return output, usage


@retry_with_backoff(max_retries=20)
async def _get_next_structure_copilot(
    structure: type[BMType],  # type[T]
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
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

    response = await copilot_client.chat.completions.create(
        model=model_id, messages=messages, response_format=response_format
    )
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from Copilot model")
    output = structure.model_validate_json(content)
    usage = TokenUsage.from_chat_completion(response.usage)
    return output, usage


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
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    use_json_object = model_id in {
        "qwen/qwen3-235b-a22b-thinking-2507",
        "z-ai/glm-4.5-air:free",
    }

    messages = update_messages_openrouter(
        messages=messages,
        structure=structure if use_json_object else None,
        use_json_object=use_json_object,
    )

    schema = structure.model_json_schema()

    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

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

    extra_body: dict[str, T.Any] = {}

    if model_id == "qwen/qwen3-235b-a22b-thinking-2507":
        extra_body["provider"] = {
            "order": ["Novita"],
            "allow_fallbacks": True,
        }
    elif model_id == "qwen/qwen3-235b-a22b":
        extra_body["provider"] = {
            "only": ["cerebras"],
        }
    elif model_id == "google/gemini-3-pro-preview":
        extra_body["provider"] = {
            "only": ["Google"],
            "allow_fallbacks": False,
        }
        extra_body["reasoning"] = {"effort": "high"}

    response = await openrouter_client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format=response_format,
        max_tokens=100_000,
        extra_body=extra_body,
    )

    if not response.choices:
        raise Exception(f"OpenRouter returned no choices: {response}")
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from OpenRouter model")

    output = structure.model_validate_json(content)
    return output, TokenUsage.from_chat_completion(response.usage)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_groq(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
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
    response = await groq_client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format=response_format,
        max_tokens=50_000,
    )
    if not response.choices:
        raise Exception(f"Groq returned no choices: {response}")
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from Groq model")
    output = structure.model_validate_json(content)
    return output, TokenUsage.from_chat_completion(response.usage)


def _log_completion_parse_failure(
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
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    messages = update_messages_openrouter(messages=messages)
    api_model = model_id

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

    if not response.choices:
        _log_completion_parse_failure(
            reason="no_choices",
            response=response,
            api_model=api_model,
            structure_name=structure.__name__,
        )
        raise Exception(f"{model_id} returned no choices: {response}")
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

    if not content:
        _log_completion_parse_failure(
            reason="empty_or_missing_message_content",
            response=response,
            api_model=api_model,
            structure_name=structure.__name__,
        )
        raise Exception(f"Empty response from {model_id} model")

    output = structure.model_validate_json(content)
    return output, TokenUsage.from_chat_completion(response.usage)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_kilo(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
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
        model=model_id,
        messages=messages,
        response_format=response_format,
    )
    if not response.choices:
        raise Exception(f"Kilo returned no choices: {response}")
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from Kilo model")

    output = structure.model_validate_json(content)
    return output, TokenUsage.from_chat_completion(response.usage)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_gemini(
    structure: type[BMType],  # type[T]
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    prompt = update_messages_gemini(messages=messages)

    config = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=structure,
    )

    config.thinking_config = ThinkingConfig(thinking_level=ThinkingLevel.HIGH)

    response = await gemini_client.aio.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )

    usage_metadata = response.usage_metadata
    token_usage = TokenUsage.from_gemini_metadata(usage_metadata)

    if hasattr(response, "parsed") and response.parsed:
        return T.cast(BMType, response.parsed), token_usage

    content = response.text
    if not content:
        raise Exception("Empty response from Gemini model")

    output = structure.model_validate_json(content)
    return output, token_usage


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


@retry_with_backoff(max_retries=20)
async def _get_next_structure_pydantic_gateway(
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> tuple[BMType, TokenUsage]:
    """
    Use Pydantic AI Gateway to call Gemini models with native thinking_config support.
    Requires PYDANTIC_AI_GATEWAY_API_KEY environment variable.
    """

    settings = GoogleModelSettings(
        google_thinking_config=ThinkingConfigDict(thinking_level=ThinkingLevel.HIGH),
    )

    model_name = model_id.split(":")[-1]
    google_model = GoogleModel(model_name, provider=gateway_client)

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

    pyd_usage = result.usage()
    token_usage = TokenUsage.from_pydantic_ai_usage(pyd_usage)
    return result.output, token_usage


def _run_provider_structure_function(
    provider: str,
    structure: type[BMType],
    model_id: str,
    messages: list,
) -> T.Awaitable[tuple[BMType, TokenUsage]]:
    return {
        "openai": _get_next_structure_openai,
        "copilot": _get_next_structure_copilot,
        "anthropic": _get_next_structure_anthropic,
        "xai": _get_next_structure_xai,
        "deepseek": _get_next_structure_deepseek,
        "google": _get_next_structure_gemini,
        "gateway": _get_next_structure_pydantic_gateway,
        "kilo": _get_next_structure_kilo,
        "lmstudio": _get_next_structure_lmstudio,
        "openrouter": _get_next_structure_openrouter,
        "groq": _get_next_structure_groq,
    }[provider](structure, model_id, messages)


async def get_next_structure(
    structure: type[BMType],
    llm: str,
    messages: list,
) -> BMType:
    provider, model_id = parse_llm(llm)
    async with API_SEMAPHORE:
        start = time.time()
        try:
            result, token_usage = await _run_provider_structure_function(
                provider,
                structure,
                model_id,
                messages,
            )
        except KeyError as e:
            raise KeyError(f"Provider {provider} not found") from e
        raw_response = {"response": "hidden"}
        duration = time.time() - start
        if os.getenv("LOG_LEVEL", "INFO") == "DEBUG":
            raw_response = result.model_dump()
        log.info(
            "API call completed",
            duration_seconds=duration,
            token_usage=token_usage.model_dump(),
            response=raw_response,
        )

        record_llm_token_usage(token_usage)
        return result
