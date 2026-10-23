"""
Cost tracking for PydanticAI agents.

Wraps Agent.run so every agent run (the SQL agent AND the judge) records
its token usage. run_sync calls run internally, so both are covered.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse

# USD per 1M tokens: (input, output). Check https://openai.com/api/pricing
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


@dataclass
class UsageTracker:
    # model name -> [input_tokens, output_tokens, requests]
    by_model: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))
    # test name -> cost in USD
    by_test: dict = field(default_factory=lambda: defaultdict(float))
    current_test: str | None = None

    def record(self, model_name, input_tokens, output_tokens):
        stats = self.by_model[model_name]
        stats[0] += input_tokens
        stats[1] += output_tokens
        stats[2] += 1
        if self.current_test:
            self.by_test[self.current_test] += price(model_name, input_tokens, output_tokens)

    def total_cost(self):
        return sum(price(m, i, o) for m, (i, o, _) in self.by_model.items())

    def report(self):
        lines = []
        for model, (i, o, n) in self.by_model.items():
            lines.append(
                f"{model}: {n} requests, {i:,} input + {o:,} output tokens"
                f" = ${price(model, i, o):.4f}"
            )
        lines.append(f"TOTAL: ${self.total_cost():.4f}")
        return "\n".join(lines)


def price(model_name, input_tokens, output_tokens):
    # OpenAI returns versioned names like "gpt-4o-mini-2024-07-18"
    for known in sorted(PRICES, key=len, reverse=True):
        if model_name and model_name.startswith(known):
            input_price, output_price = PRICES[known]
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return 0.0


tracker = UsageTracker()
_original_run = Agent.run
_patched = False


async def _run_with_tracking(self, *args, **kwargs):
    result = await _original_run(self, *args, **kwargs)
    for message in result.new_messages():
        if isinstance(message, ModelResponse):
            tracker.record(
                message.model_name or "unknown",
                message.usage.input_tokens,
                message.usage.output_tokens,
            )
    return result


def install():
    """Patch Agent.run once. Safe to call more than once."""
    global _patched
    if not _patched:
        Agent.run = _run_with_tracking
        _patched = True
