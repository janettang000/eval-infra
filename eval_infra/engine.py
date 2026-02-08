from __future__ import annotations

from typing import Any

import sglang as sgl
from transformers import AutoTokenizer


class Engine:
    """Thin wrapper around sglang.Engine for batch inference."""

    def __init__(self, model_path: str, **engine_kwargs: Any):
        self.model_path = model_path
        self._engine = sgl.Engine(model_path=model_path, **engine_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        **kwargs: Any,
    ) -> list[str]:
        """Run batch text completion on raw string prompts."""
        sampling_params = {"temperature": temperature, "max_new_tokens": max_new_tokens, **kwargs}
        outputs = self._engine.generate(prompts, sampling_params)
        return [o["text"] for o in outputs]

    def chat(
        self,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        **kwargs: Any,
    ) -> list[str]:
        """Run batch chat completion, applying the model's chat template."""
        prompts = [
            self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]
        return self.generate(prompts, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)

    def shutdown(self) -> None:
        self._engine.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
