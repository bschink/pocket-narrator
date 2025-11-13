import os
import pytest
from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.tokenizers.base_tokenizer import AbstractTokenizer
from pocket_narrator.tokenizers.character_tokenizer import CharacterTokenizer
from pocket_narrator.tokenizers.bpe_tokenizer import BPETokenizer

DEFAULT_SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

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
    assert tokenizer.unk_token_id == 1

def test_untrained_tokenizer_raises_runtime_error():
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
    text = "a test"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text)) == text

def test_encode_decode_after_training_with_unknowns():
    tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    tokenizer.train(["abc"])
    text_with_unknowns = "ad"
    encoded = tokenizer.encode(text_with_unknowns)
    decoded = tokenizer.decode(encoded)
    unk_id = tokenizer.unk_token_id
    assert encoded == [tokenizer.char_to_idx['a'], unk_id]
    assert decoded == "a<unk>"

# --- Tests for the BPETokenizer Class ---

def test_bpe_tokenizer_initialization():
    tokenizer = BPETokenizer(vocab_size=512, special_tokens=DEFAULT_SPECIAL_TOKENS)
    assert tokenizer.vocab_size == 512
    assert len(tokenizer.vocab) == 256
    for special_token, idx in DEFAULT_SPECIAL_TOKENS.items():
        assert idx in tokenizer.vocab
        assert tokenizer.vocab[idx] == special_token.encode('utf-8')

def test_bpe_initialization_fails_with_small_vocab():
    with pytest.raises(ValueError, match="Vocab size must be at least 256"):
        BPETokenizer(vocab_size=100, special_tokens=DEFAULT_SPECIAL_TOKENS)

def test_bpe_train_creates_merges():
    corpus = ["aaaaa"]
    tokenizer = BPETokenizer(vocab_size=257, special_tokens={})
    tokenizer.train(corpus)
    assert tokenizer.get_vocab_size() == 257
    most_frequent_pair = (97, 97)
    assert len(tokenizer.merges) == 1
    assert most_frequent_pair in tokenizer.merges
    assert tokenizer.merges[most_frequent_pair] == 256

def test_bpe_encode_decode_roundtrip():
    corpus = ["a simple sentence for testing"]
    tokenizer = BPETokenizer(vocab_size=300, special_tokens={})
    tokenizer.train(corpus)
    text = "this is a test sentence"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_bpe_save_and_load_roundtrip(tmp_path):
    corpus = ["a simple sentence for testing the save and load functionality"]
    original_tokenizer = BPETokenizer(vocab_size=300, special_tokens=DEFAULT_SPECIAL_TOKENS)
    original_tokenizer.train(corpus)
    save_dir = tmp_path / "bpe_tokenizer"
    original_tokenizer.save(save_dir)
    loaded_tokenizer = BPETokenizer.load(save_dir)
    assert isinstance(loaded_tokenizer, BPETokenizer)
    assert loaded_tokenizer.merges == original_tokenizer.merges
    assert loaded_tokenizer.special_tokens == original_tokenizer.special_tokens
    text = "test functionality"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text)) == text

def test_bpe_special_tokens_handling():
    corpus = ["some text"]
    special_token = "<|endoftext|>"
    special_tokens = {special_token: 300}
    tokenizer = BPETokenizer(vocab_size=300, special_tokens=special_tokens)
    tokenizer.train(corpus)
    text_with_special = f"some text {special_token}"
    with pytest.raises(AssertionError):
        tokenizer.encode(text_with_special)
    encoded = tokenizer._encode_internal(text_with_special, allowed_special="all")
    assert encoded[-1] == 300
    decoded = tokenizer.decode(encoded)
    assert decoded == text_with_special

# --- Tests for the get_tokenizer Factory Function ---

def test_get_tokenizer_loads_existing_file(tmp_path):
    corpus1 = ["hello"]
    corpus2 = ["world"]
    tokenizer_dir = tmp_path / "char_tokenizer_for_loading"
    initial_tokenizer = CharacterTokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
    initial_tokenizer.train(corpus1)
    initial_tokenizer.save(tokenizer_dir)
    loaded_tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_dir,
        train_corpus=corpus2
    )
    assert isinstance(loaded_tokenizer, AbstractTokenizer)
    assert loaded_tokenizer.get_vocab_size() == len(DEFAULT_SPECIAL_TOKENS) + 4

def test_get_tokenizer_trains_and_saves_if_nonexistent(tmp_path):
    corpus = ["new tokenizer"]
    tokenizer_dir = tmp_path / "char_tokenizer_for_saving"
    tokenizer_config = {"special_tokens": DEFAULT_SPECIAL_TOKENS}
    tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_dir,
        train_corpus=corpus,
        **tokenizer_config
    )
    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.get_vocab_size() > len(DEFAULT_SPECIAL_TOKENS)
    assert os.path.exists(os.path.join(tokenizer_dir, "vocab.json"))

def test_get_tokenizer_handles_bpe_type(tmp_path):
    tokenizer_config = {"vocab_size": 260, "special_tokens": DEFAULT_SPECIAL_TOKENS}
    tokenizer_dir = tmp_path / "bpe_tokenizer"
    tokenizer = get_tokenizer(
        tokenizer_type="bpe",
        train_corpus=["abc"],
        tokenizer_path=tokenizer_dir,
        **tokenizer_config
    )
    assert isinstance(tokenizer, BPETokenizer)
    assert tokenizer.vocab_size >= 260
    assert os.path.exists(os.path.join(tokenizer_dir, "bpe.model"))

def test_get_tokenizer_raises_error_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        get_tokenizer(tokenizer_type="some_future_tokenizer", train_corpus=["abc"])

def test_get_tokenizer_raises_error_when_no_path_or_corpus():
    with pytest.raises(ValueError, match="Must provide either a valid tokenizer_path or a train_corpus"):
        get_tokenizer(tokenizer_type="character")