# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer


class TikTokenTokenizer(PreTrainedTokenizer):
    """Adapter for TikToken tokenizers in vLLM"""

    def __init__(self, tokenizer_name: str, **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                        trust_remote_code=True,
                                                        **kwargs)

        self.bos_token = self._tokenizer.bos_token
        self.eos_token = self._tokenizer.eos_token
        self.pad_token = self._tokenizer.pad_token
        self.unk_token = self._tokenizer.unk_token

        if hasattr(self._tokenizer, "special_tokens"):
            map_fn = lambda x: self._tokenizer.special_tokens[x]
        elif hasattr(self._tokenizer, "convert_tokens_to_ids"):
            map_fn = lambda x: self._tokenizer.convert_tokens_to_ids(x)
        else:
            raise ValueError(
                f"Invalid tokenizer type: {type(self._tokenizer)}")

        self.bos_token_id = map_fn(self.bos_token)
        self.eos_token_id = map_fn(self.eos_token)
        self.pad_token_id = map_fn(self.pad_token)
        self.unk_token_id = map_fn(self.unk_token)

        self.vocab_size = len(self._tokenizer.get_vocab())
        self.model_max_length = 131072

        # Build reverse vocab for convert_ids_to_tokens
        self._id_to_token = {v: k for k, v in self._tokenizer.get_vocab().items()}

    def convert_ids_to_tokens(
        self,
        ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
    ) -> Union[str, list[str]]:
        """Convert token IDs to tokens.

        This method is required for incremental detokenization in vLLM.
        """
        if isinstance(ids, int):
            # Single token ID
            token = self._id_to_token.get(ids)
            if token is None:
                # Fallback: decode the single token
                token = self._tokenizer.decode([ids])
            return token

        # List of token IDs
        tokens = []
        for token_id in ids:
            token = self._id_to_token.get(token_id)
            if token is None:
                # Fallback: decode the single token
                token = self._tokenizer.decode([token_id])
            tokens.append(token)
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        """Convert a single token ID to its string representation."""
        token = self._id_to_token.get(index)
        if token is None:
            # Fallback: decode the single token
            token = self._tokenizer.decode([index])
        return token

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
