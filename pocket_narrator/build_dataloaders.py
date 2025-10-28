# Here’s a drop-in implementation for build_dataloaders that keeps everything pure-Python, uses a simple character-level tokenizer
# with an explicit <eos> token between lines, and yields (x, y) batches of integer IDs for next-token prediction.


def build_dataloaders(
    data_dir: str,
    split_ratio: float = 0.98,
    block_size: int = 256,
    batch_size: int = 32,
    seed: int = 1337
):
    """
    Returns: train_loader, val_loader, vocab_size, eos_token_id, decode_fn
    - train/val loaders yield lists of length `batch_size`, each item a list[int] of length `block_size` (x) and same for y shifted by 1.
      Concretely: each iterator yields tuples (X_batch, Y_batch) where both are List[List[int]] of shape [B, T].
    - eos_token is inserted between lines.
    - decode_fn: List[int] -> str (drops <eos> when decoding).
    """
    # 1) Find a text file inside data_dir
    candidate_files = ["train.txt", "data.txt", "corpus.txt", "text.txt"]
    dataset_path = None
    for name in candidate_files:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            dataset_path = p
            break
    if dataset_path is None:
        raise FileNotFoundError(
            f"No dataset file found in {data_dir}. Expected one of: {', '.join(candidate_files)}"
        )

    # 2) Load lines and split
    lines = load_text_dataset(dataset_path)
    train_lines, val_lines = split_text(lines, val_ratio=1.0 - split_ratio, seed=seed)

    # 3) Build char-level vocabulary with a dedicated <eos> token
    EOS = "<eos>"
    # union of characters from *all* data so val encoding never hits OOV
    charset = sorted({ch for ln in lines for ch in ln})
    itos = [EOS] + charset  # put EOS at index 0 for convenience
    stoi = {ch: i for i, ch in enumerate(itos)}
    eos_token_id = stoi[EOS]
    vocab_size = len(itos)

    def encode(s: str) -> List[int]:
        # map characters to ids (no OOV possible because vocab built on full corpus)
        return [stoi[ch] for ch in s]

    def decode(ids: List[int]) -> str:
        # drop EOS tokens when decoding for clean text
        return "".join(itos[i] for i in ids if i != eos_token_id)

    # 4) Turn a list of lines into a single token stream with EOS between lines
    def lines_to_tokens(raw_lines: List[str]) -> List[int]:
        toks: List[int] = []
        for i, ln in enumerate(raw_lines):
            toks.extend(encode(ln))
            toks.append(eos_token_id)  # boundary between samples
        return toks

    train_tokens = lines_to_tokens(train_lines)
    val_tokens = lines_to_tokens(val_lines)

    # 5) Chunk continuous token stream into (x, y) sequences for next-token prediction
    # We use non-overlapping contiguous blocks for simplicity & determinism.
    def stream_to_xy_blocks(tokens: List[int], T: int) -> Tuple[List[List[int]], List[List[int]]]:
        X, Y = [], []
        # ensure we have at least T+1 tokens to make one x/y pair
        for i in range(0, max(0, len(tokens) - (T + 1) + 1), T):
            x = tokens[i:i + T]
            y = tokens[i + 1:i + 1 + T]
            if len(x) == T and len(y) == T:
                X.append(x)
                Y.append(y)
        return X, Y

    train_X, train_Y = stream_to_xy_blocks(train_tokens, block_size)
    val_X, val_Y = stream_to_xy_blocks(val_tokens, block_size)

    # 6) Simple deterministic batchers
    rng = random.Random(seed)

    def make_loader(X: List[List[int]], Y: List[List[int]], shuffle: bool = True):
        idxs = list(range(len(X)))
        if shuffle:
            rng.shuffle(idxs)

        def iterator():
            for s in range(0, len(idxs), batch_size):
                sel = idxs[s:s + batch_size]
                if not sel:
                    break
                Xb = [X[i] for i in sel]
                Yb = [Y[i] for i in sel]
                # drop last incomplete batch for consistency with typical training loops
                if len(Xb) == batch_size:
                    yield Xb, Yb
        return iterator()

    train_loader = make_loader(train_X, train_Y, shuffle=True)
    val_loader = make_loader(val_X, val_Y, shuffle=False)

    # 7) Provide a decode function that callers can use without tokenizer.py
    decode_fn = decode

    return train_loader, val_loader, vocab_size, eos_token_id, decode_fn
