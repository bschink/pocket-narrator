import os
import pytest
from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.tokenizers.base_tokenizer import AbstractTokenizer
from pocket_narrator.tokenizers.character_tokenizer import CharacterTokenizer
from pocket_narrator.tokenizers.bpe_tokenizer import BPETokenizer

DEFAULT_SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

# --- Tests for the CharacterTokenizer Class ---

def test_character_tokenizer_initialization():
    tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    assert tokenizer.get_vocab_size() == 0
    assert tokenizer.vocabulary == []
    assert not tokenizer.char_to_idx
    assert tokenizer.unk_token_id is None

def test_character_train_method_builds_vocab_correctly():
    corpus = ["hello", "world"]
    tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    tokenizer.train(corpus)
    unique_chars = sorted(list(set("".join(corpus))))
    assert tokenizer.get_vocab_size() == len(DEFAULT_SPECIAL_TOKENS) + len(unique_chars)
    assert tokenizer.unk_token_id == tokenizer.token_to_id("<|unk|>")

def test_character_untrained_tokenizer_raises_runtime_error():
    tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.encode("hello")
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.decode([1, 2, 3])

def test_character_save_and_load_roundtrip(tmp_path):
    corpus = ["a simple test!"]
    original_tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    original_tokenizer.train(corpus)
    save_dir = tmp_path / "char_tokenizer_test"
    original_tokenizer.save(save_dir)
    loaded_tokenizer = CharacterTokenizer.load(save_dir)
    assert loaded_tokenizer.vocabulary == original_tokenizer.vocabulary
    assert loaded_tokenizer.char_to_idx == original_tokenizer.char_to_idx
    assert loaded_tokenizer.special_tokens == {token: loaded_tokenizer.char_to_idx[token] for token in DEFAULT_SPECIAL_TOKENS}
    text = "a test"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text)) == text

def test_character_encode_decode_after_training_with_unknowns():
    tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    tokenizer.train(["abc"])
    text_with_unknowns = "ad"
    encoded = tokenizer.encode(text_with_unknowns)
    decoded = tokenizer.decode(encoded)
    unk_id = tokenizer.unk_token_id
    assert encoded == [tokenizer.char_to_idx['a'], unk_id]
    assert decoded == "a<|unk|>"

# --- Tests for the BPETokenizer Class ---

def test_bpe_tokenizer_initialization():
    tokenizer = BPETokenizer(vocab_size=512, special_tokens=DEFAULT_SPECIAL_TOKENS)
    assert tokenizer.vocab_size == 512
    assert len(tokenizer.vocab) == 256
    assert tokenizer.special_token_names == DEFAULT_SPECIAL_TOKENS

def test_bpe_initialization_fails_with_small_vocab():
    with pytest.raises(ValueError, match="Vocab size must be at least 256"):
        BPETokenizer(vocab_size=100, special_tokens=DEFAULT_SPECIAL_TOKENS)

def test_bpe_train_creates_merges():
    corpus_iterator = ["aaaaa"]
    tokenizer = BPETokenizer(vocab_size=257, special_tokens={})
    tokenizer.train(corpus_iterator)
    assert tokenizer.get_vocab_size() == 257
    most_frequent_pair = (97, 97)
    assert len(tokenizer.merges) == 1
    assert most_frequent_pair in tokenizer.merges
    assert tokenizer.merges[most_frequent_pair] == 256

def test_bpe_encode_decode_roundtrip():
    corpus_iterator = ["a simple sentence for testing"]
    tokenizer = BPETokenizer(vocab_size=300, special_tokens={})
    tokenizer.train(corpus_iterator)
    text = "this is a test sentence"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_bpe_save_and_load_roundtrip(tmp_path):
    corpus_iterator = ["a simple sentence for testing the save and load functionality"]
    original_tokenizer = BPETokenizer(vocab_size=300, special_tokens=DEFAULT_SPECIAL_TOKENS)
    original_tokenizer.train(corpus_iterator)
    save_dir = tmp_path / "bpe_tokenizer"
    original_tokenizer.save(save_dir)
    loaded_tokenizer = BPETokenizer.load(save_dir)
    assert isinstance(loaded_tokenizer, BPETokenizer)
    assert loaded_tokenizer.merges == original_tokenizer.merges
    assert loaded_tokenizer.special_tokens == original_tokenizer.special_tokens
    text = "test functionality"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text)) == text

def test_bpe_special_tokens_handling():
    corpus_iterator = ["some text"]
    special_token = "<|endoftext|>"
    special_tokens = [special_token]
    tokenizer = BPETokenizer(vocab_size=301, special_tokens=special_tokens)
    tokenizer.train(corpus_iterator)
    text_with_special = f"some text {special_token}"
    # encode() allows all special tokens by default
    encoded = tokenizer.encode(text_with_special)
    assert tokenizer.special_tokens[special_token] in encoded
    decoded = tokenizer.decode(encoded)
    assert decoded == text_with_special

# --- Tests for the get_tokenizer Factory Function ---

def test_get_tokenizer_loads_existing(tmp_path):
    corpus = ["hello"]
    tokenizer_dir = tmp_path / "char_tokenizer_for_loading"
    initial_tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    initial_tokenizer.train(corpus)
    initial_tokenizer.save(tokenizer_dir)
    loaded_tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_dir,
    )
    assert isinstance(loaded_tokenizer, CharacterTokenizer)
    assert loaded_tokenizer.get_vocab_size() == len(DEFAULT_SPECIAL_TOKENS) + 4

def test_get_tokenizer_instantiates_new_if_nonexistent(tmp_path):
    tokenizer_dir = tmp_path / "new_tokenizer"
    tokenizer_config = {"special_tokens": DEFAULT_SPECIAL_TOKENS}
    tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_dir,
        **tokenizer_config
    )
    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.get_vocab_size() == 0

def test_get_tokenizer_passes_kwargs_to_bpe(tmp_path):
    tokenizer_config = {"vocab_size": 260, "special_tokens": DEFAULT_SPECIAL_TOKENS}
    tokenizer_dir = tmp_path / "bpe_tokenizer"
    tokenizer = get_tokenizer(
        tokenizer_type="bpe",
        tokenizer_path=tokenizer_dir,
        **tokenizer_config
    )
    assert isinstance(tokenizer, BPETokenizer)
    assert tokenizer.vocab_size == 260

def test_get_tokenizer_raises_error_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        get_tokenizer(tokenizer_type="some_future_tokenizer")