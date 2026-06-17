"""
Tests for nanochat.tokenizer - BPE tokenizer implementations.

Run: python -m pytest tests/test_tokenizer.py -v
"""

import pytest
from nanochat.tokenizer import (
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    RustBPETokenizer,
)


class TestSpecialTokens:
    """Test special token definitions."""

    def test_bos_is_first(self):
        assert SPECIAL_TOKENS[0] == "<|bos|>"

    def test_all_special_tokens_have_delimiters(self):
        for token in SPECIAL_TOKENS:
            assert token.startswith("<|") and token.endswith("|>")

    def test_expected_tokens_present(self):
        expected = [
            "<|bos|>", "<|user_start|>", "<|user_end|>",
            "<|assistant_start|>", "<|assistant_end|>",
            "<|python_start|>", "<|python_end|>",
            "<|output_start|>", "<|output_end|>",
        ]
        for token in expected:
            assert token in SPECIAL_TOKENS


class TestRustBPETokenizerPretrained:
    """Test RustBPETokenizer using GPT-2 pretrained encoding."""

    @pytest.fixture
    def tokenizer(self):
        return RustBPETokenizer.from_pretrained("gpt2")

    def test_vocab_size(self, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        # GPT-2 has 50257 tokens
        assert vocab_size == 50257

    def test_encode_simple(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_roundtrip(self, tokenizer):
        text = "Hello, world!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_encode_with_prepend(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        ids = tokenizer.encode("hello", prepend=bos)
        assert ids[0] == bos

    def test_encode_with_append(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        ids = tokenizer.encode("hello", append=bos)
        assert ids[-1] == bos

    def test_encode_batch(self, tokenizer):
        texts = ["hello", "world", "test"]
        ids_list = tokenizer.encode(texts)
        assert isinstance(ids_list, list)
        assert len(ids_list) == 3
        for ids in ids_list:
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_encode_batch_with_prepend(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        texts = ["hello", "world"]
        ids_list = tokenizer.encode(texts, prepend=bos)
        for ids in ids_list:
            assert ids[0] == bos

    def test_get_bos_token_id(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        assert isinstance(bos, int)
        assert bos >= 0

    def test_encode_special(self, tokenizer):
        # GPT-2 uses <|endoftext|> as BOS
        eot_id = tokenizer.encode_special("<|endoftext|>")
        assert isinstance(eot_id, int)
        assert eot_id == 50256  # Known GPT-2 value

    def test_callable(self, tokenizer):
        # Tokenizer should be callable (same as encode)
        ids1 = tokenizer.encode("hello")
        ids2 = tokenizer("hello")
        assert ids1 == ids2

    def test_empty_string(self, tokenizer):
        ids = tokenizer.encode("")
        assert ids == []

    def test_encode_invalid_type(self, tokenizer):
        with pytest.raises(ValueError, match="Invalid input type"):
            tokenizer.encode(42)

    def test_special_characters(self, tokenizer):
        text = "Hello! @#$% 123"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_unicode(self, tokenizer):
        text = "café résumé"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text


class TestRustBPETokenizerRenderConversation:
    """Test render_conversation method using a trained tokenizer with special tokens."""

    @pytest.fixture
    def tokenizer(self):
        # Train a small tokenizer that includes nanochat's special tokens
        texts = ["hello world " * 100, "the quick brown fox " * 100, "testing one two three " * 100]
        tok = RustBPETokenizer.train_from_iterator(iter(texts), vocab_size=300)
        return tok

    def test_simple_user_assistant(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert len(ids) == len(mask)
        assert len(ids) > 0
        # BOS token should be first
        assert ids[0] == tokenizer.get_bos_token_id()
        # BOS should be masked (0)
        assert mask[0] == 0
        # Some tokens should be supervised (mask=1) for assistant content
        assert 1 in mask

    def test_system_message_merged(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert len(ids) == len(mask)
        assert len(ids) > 0

    def test_max_tokens_truncation(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Tell me a very long story " * 100},
                {"role": "assistant", "content": "Once upon a time " * 100},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation, max_tokens=50)
        assert len(ids) <= 50
        assert len(mask) <= 50

    def test_assistant_parts_with_tool_calls(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Calculate 2+2"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me calculate that."},
                    {"type": "python", "text": "print(2+2)"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "The answer is 4."},
                ]},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert len(ids) == len(mask)
        # python_output should be masked (0) since it comes from Python at test time
        # At least some tokens should be supervised
        assert 1 in mask

    def test_render_for_completion(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        ids = tokenizer.render_for_completion(conversation)
        assert isinstance(ids, list)
        assert len(ids) > 0
        # Should end with assistant_start token
        # (last message popped, then assistant_start appended)

    def test_visualize_tokenization(self, tokenizer):
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        viz = tokenizer.visualize_tokenization(ids, mask)
        assert isinstance(viz, str)
        assert len(viz) > 0
