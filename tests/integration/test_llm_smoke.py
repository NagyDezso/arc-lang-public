from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from src.llms.messages import get_next_message_openai
from src.llms.structured import _get_next_structure_openrouter


@pytest.mark.integration
async def test_get_next_message_openai_smoke() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    model = os.environ.get("OPENAI_SMOKE_MODEL", "gpt-4o-mini")
    text = await get_next_message_openai(
        model_id=model,
        inputs=[{"role": "user", "content": "Reply with exactly: ok"}],
    )
    assert isinstance(text, str)
    assert text.strip()


@pytest.mark.integration
async def test_openrouter_structured_smoke() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    class Reasoning(BaseModel):
        reasoning: str = Field(..., description="Reasoning for the math problem.")
        answer: int = Field(..., description="The answer to the math problem.")

    response, _usage = await _get_next_structure_openrouter(
        structure=Reasoning,
        model_id="openai/gpt-oss-120b:free",
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
    assert isinstance(response.answer, int)
