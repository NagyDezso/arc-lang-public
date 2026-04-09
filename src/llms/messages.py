import os
import typing as T

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.llms.models import (
    ANTHROPIC_MODEL_CONFIG,
    DEFAULT_ANTHROPIC_MODEL_CONFIG,
    LMSTUDIO_OPENAI_BASE_URL,
    parse_llm,
)
from src.llms.openai_responses import (
    OPENAI_MODEL_MAX_OUTPUT_TOKENS,
    create_and_poll_response,
)

COPILOT_BASE_URL = "http://localhost:4141/v1"


class GridOutput(BaseModel):
    grid: list[list[int]] = Field(..., description="Extracted 2D grid of integers")


def _extract_output_text(response: T.Any) -> str:
    if hasattr(response, "output_text"):
        output_text = response.output_text
        if isinstance(output_text, str) and output_text.strip():
            return output_text

    payload: dict[str, T.Any]
    if isinstance(response, dict):
        payload = response
    elif hasattr(response, "model_dump"):
        payload = T.cast(dict[str, T.Any], response.model_dump())
    else:
        raise ValueError("Unable to interpret OpenAI response payload.")

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    outputs = payload.get("output") or []
    texts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        contents = item.get("content") or []
        for content in contents:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    if texts:
        return "\n".join(texts)

    raise ValueError("Unable to extract text content from OpenAI response output.")


async def extract_grid_from_text(
    *,
    llm: str,
    text: str,
) -> list[list[int]]:
    provider, model_id = parse_llm(llm)
    if provider != "openai":
        raise ValueError(f"extract_grid_from_text only supports openai/*, got {llm!r}")
    timeout = 10_800 if model_id in {"gpt-5.2", "gpt-5-pro"} else 120
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=10
    )

    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": text}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_grid",
                    "description": "Extract the final 2D integer grid from the given text. The response may contain many 2d integer grids but extract the final answer from the text, which is a 2d list of integers.",
                    "parameters": GridOutput.model_json_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_grid"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    grid_data = GridOutput.model_validate_json(tool_call.function.arguments)
    return grid_data.grid


async def get_next_message_openai(model_id: str, inputs: list[dict[str, str]]) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=10_800, max_retries=10
    )
    params = {}
    model_name = model_id

    if model_id.endswith("_high"):
        params["reasoning"] = {"effort": "high"}
        model_name = model_id.replace("_high", "")

    max_output_tokens = OPENAI_MODEL_MAX_OUTPUT_TOKENS.get(model_name, 100_000)
    body: dict[str, T.Any] = {
        "model": model_name,
        "input": inputs,
        "max_output_tokens": max_output_tokens,
    }
    body.update(params)

    response = await create_and_poll_response(
        client,
        model_id=model_name,
        create_kwargs=body,
    )
    return _extract_output_text(response)


async def get_next_message_openrouter(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=500,
        max_retries=10,
        base_url="https://openrouter.ai/api/v1",
    )
    completion = await client.chat.completions.create(
        model=model_id,
        max_tokens=50_000,
        max_completion_tokens=50_000,
        messages=inputs,
        temperature=1,
    )
    return completion.choices[0].message.content


async def get_next_message_copilot(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ.get("COPILOT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "copilot",
        timeout=500,
        max_retries=10,
        base_url=COPILOT_BASE_URL,
    )
    completion = await client.chat.completions.create(
        model=model_id,
        max_tokens=50_000,
        messages=inputs,
        temperature=1,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError(f"Empty response from Copilot model {model_id}")
    return content


async def get_next_message_deepseek(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=300,
        max_retries=10,
        base_url="https://api.deepseek.com",
    )
    response = await client.chat.completions.create(
        model=model_id,
        max_tokens=8192,
        messages=inputs,
    )
    return response.choices[0].message.content


async def get_next_message_lmstudio(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
        timeout=10_800,
        max_retries=10,
        base_url=LMSTUDIO_OPENAI_BASE_URL,
    )
    completion = await client.chat.completions.create(
        model=model_id,
        messages=inputs,
        temperature=1,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError(f"Empty response from LM Studio model {model_id}")
    return content


async def get_next_message_kilo(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["KILO_API_KEY"],
        timeout=500,
        max_retries=10,
        base_url="https://api.kilo.ai/api/gateway",
    )
    completion = await client.chat.completions.create(
        model=model_id,
        max_tokens=50_000,
        messages=inputs,
        temperature=1,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError(f"Empty response from Kilo model {model_id}")
    return content


async def get_next_message_anthropic(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    config = ANTHROPIC_MODEL_CONFIG.get(model_id, DEFAULT_ANTHROPIC_MODEL_CONFIG)
    client = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300, max_retries=30
    )

    new_inputs = []
    for _input in inputs:
        new_inputs.append(
            {
                "role": _input["role"],
                "content": [
                    {
                        "type": "text",
                        "text": _input["content"],
                    }
                ],
            }
        )
    new_inputs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    params = {}
    if config.max_thinking_tokens:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.max_thinking_tokens,
        }

    message = await client.messages.create(
        model=model_id,
        max_tokens=config.max_tokens,
        messages=new_inputs,
        **params,
    )
    return message.content[-1].text


async def get_next_message_gemini(
    model_id: str,
    inputs: list[dict[str, str]],
) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    contents = []
    for i in inputs:
        role = i["role"]
        if role == "assistant":
            role = "model"
        contents.append(
            genai.types.ContentDict(
                role=role, parts=[genai.types.PartDict(text=i["content"])]
            ),
        )

    config: dict[str, T.Any] = {
        "max_output_tokens": 65_536,
    }

    if model_id == "gemini-3-pro-preview":
        config["thinking_config"] = {
            "thinking_budget": 65_535,
        }

    response = await client.aio.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )

    return response.text
