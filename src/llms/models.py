from __future__ import annotations

from anthropic.types import Usage
from google.genai.types import GenerateContentResponseUsageMetadata
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import ResponseUsage
from pydantic import BaseModel
from pydantic_ai import RunUsage
from xai_sdk.proto.v6.usage_pb2 import SamplingUsage


def parse_llm(llm: str) -> tuple[str, str]:
    """
    Split `provider/model_id` on the first `/` only.
    Example: openrouter/qwen/qwen3.5 -> ("openrouter", "qwen/qwen3.5")
    """
    llm = llm.strip()
    if "/" not in llm:
        raise ValueError(
            f"Invalid llm {llm!r}: expected 'provider/model_id' (at least one '/')"
        )
    provider, model_id = llm.split("/", 1)
    if not provider or not model_id:
        raise ValueError(f"Invalid llm {llm!r}: empty provider or model_id")
    return provider, model_id


class ModelPricing(BaseModel):
    prompt_tokens: float
    reasoning_tokens: float
    completion_tokens: float


MODEL_PRICING_D: dict[str, ModelPricing] = {
    "xai/grok-4": ModelPricing(
        prompt_tokens=300 / 1_000_000,
        reasoning_tokens=1_500 / 1_000_000,
        completion_tokens=1_500 / 1_000_000,
    ),
    "xai/grok-3-mini-fast": ModelPricing(
        prompt_tokens=60 / 1_000_000,
        reasoning_tokens=400 / 1_000_000,
        completion_tokens=400 / 1_000_000,
    ),
    "openai/o3": ModelPricing(
        prompt_tokens=5_000 / 1_000_000,
        reasoning_tokens=25_000 / 1_000_000,
        completion_tokens=15_000 / 1_000_000,
    ),
    "openai/o3-pro": ModelPricing(
        prompt_tokens=1_5_00 / 1_000_000,
        reasoning_tokens=6_000 / 1_000_000,
        completion_tokens=6_000 / 1_000_000,
    ),
    "openai/o4-mini": ModelPricing(
        prompt_tokens=300 / 1_000_000,
        reasoning_tokens=1_200 / 1_000_000,
        completion_tokens=1_200 / 1_000_000,
    ),
    "openai/gpt-4.1": ModelPricing(
        prompt_tokens=250 / 1_000_000,
        reasoning_tokens=1_000 / 1_000_000,
        completion_tokens=1_000 / 1_000_000,
    ),
    "openai/gpt-4.1-mini": ModelPricing(
        prompt_tokens=150 / 1_000_000,
        reasoning_tokens=600 / 1_000_000,
        completion_tokens=600 / 1_000_000,
    ),
    "openai/gpt-5": ModelPricing(
        prompt_tokens=125 / 1_000_000,
        reasoning_tokens=1_000 / 1_000_000,
        completion_tokens=1_000 / 1_000_000,
    ),
    "openai/gpt-5.2": ModelPricing(
        prompt_tokens=175 / 1_000_000,
        reasoning_tokens=1_400 / 1_000_000,
        completion_tokens=1_400 / 1_000_000,
    ),
    "openai/gpt-5-pro": ModelPricing(
        prompt_tokens=200 / 1_000_000,
        reasoning_tokens=1_200 / 1_000_000,
        completion_tokens=1_200 / 1_000_000,
    ),
    "anthropic/claude-sonnet-4-5-20250929": ModelPricing(
        prompt_tokens=3_000 / 1_000_000,
        reasoning_tokens=15_000 / 1_000_000,
        completion_tokens=15_000 / 1_000_000,
    ),
    "gemini/gemini-2.5-pro": ModelPricing(
        prompt_tokens=1_250 / 1_000_000,
        reasoning_tokens=10_000 / 1_000_000,
        completion_tokens=10_000 / 1_000_000,
    ),
    "gemini/gemini-2.5-flash-lite": ModelPricing(
        prompt_tokens=75 / 1_000_000,
        reasoning_tokens=300 / 1_000_000,
        completion_tokens=300 / 1_000_000,
    ),
    "gemini/gemini-3-flash-preview": ModelPricing(
        prompt_tokens=2_500 / 1_000_000,
        reasoning_tokens=15_000 / 1_000_000,
        completion_tokens=15_000 / 1_000_000,
    ),
    "gemini/gemini-3-pro-preview": ModelPricing(
        prompt_tokens=2_500 / 1_000_000,
        reasoning_tokens=15_000 / 1_000_000,
        completion_tokens=15_000 / 1_000_000,
    ),
    "gateway/google-vertex:gemini-3-pro-preview": ModelPricing(
        prompt_tokens=2_500 / 1_000_000,
        reasoning_tokens=15_000 / 1_000_000,
        completion_tokens=15_000 / 1_000_000,
    ),
    "openrouter/google/gemini-3-pro-preview": ModelPricing(
        prompt_tokens=2_000 / 1_000_000,
        reasoning_tokens=0.0,
        completion_tokens=12_000 / 1_000_000,
    ),
    "lmstudio/qwen3.5-27b": ModelPricing(
        prompt_tokens=0.0,
        reasoning_tokens=0.0,
        completion_tokens=0.0,
    ),
}


def _safe_int(value: int | None) -> int:
    return value if value is not None else 0


class TokenUsage(BaseModel):
    """Unified token accounting for all LLM backends."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_responses_api(cls, usage: ResponseUsage | None) -> TokenUsage:
        if usage is None:
            return cls()
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.input_tokens_details.cached_tokens,
            cached_tokens=usage.output_tokens_details.reasoning_tokens,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def from_chat_completion(cls, usage: CompletionUsage | None) -> TokenUsage:
        if usage is None:
            return cls()
        ptd = usage.prompt_tokens_details
        cached = _safe_int(ptd.cached_tokens if ptd else None)
        ctd = usage.completion_tokens_details
        rt = _safe_int(ctd.reasoning_tokens if ctd else None)
        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            reasoning_tokens=rt,
            cached_tokens=cached,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def from_anthropic(cls, usage: Usage | None) -> TokenUsage:
        if usage is None:
            return cls()
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=_safe_int(usage.cache_creation_input_tokens),
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    @classmethod
    def from_xai_grok(cls, usage: SamplingUsage) -> TokenUsage:
        return cls(
            input_tokens=usage.prompt_text_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_prompt_text_tokens,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def from_gemini_metadata(
        cls, meta: GenerateContentResponseUsageMetadata | None
    ) -> TokenUsage:
        if meta is None:
            return cls()
        return cls(
            input_tokens=_safe_int(meta.prompt_token_count),
            output_tokens=_safe_int(meta.candidates_token_count),
            cached_tokens=_safe_int(meta.cached_content_token_count),
            total_tokens=_safe_int(meta.total_token_count),
        )

    @classmethod
    def from_pydantic_ai_usage(cls, usage: RunUsage | None) -> TokenUsage:
        if usage is None:
            return cls()
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_tokens + usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
        )

    def cost(self, llm: str) -> float:
        pricing = MODEL_PRICING_D.get(llm)
        if pricing is None:
            return 0.0
        return round(
            self.input_tokens * pricing.prompt_tokens
            + self.reasoning_tokens * pricing.reasoning_tokens
            + self.output_tokens * pricing.completion_tokens,
            2,
        )
