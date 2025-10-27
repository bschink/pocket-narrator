from pocket_narrator.data_loader import load_text_dataset, split_text, batchify_text

def test_load_text_dataset(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text("Line one\n\nLine two\n", encoding="utf-8")
    lines = load_text_dataset(str(p))
    assert lines == ["Line one", "Line two"]

def test_split_text(tmp_path):
    p = tmp_path / "many.txt"
    p.write_text("\n".join([f"L{i}" for i in range(10)]), encoding="utf-8")
    lines = load_text_dataset(str(p))
    train, val = split_text(lines, val_ratio=0.3, seed=7)
    assert len(train) == 7
    assert len(val) == 3
    # deterministic
    train2, val2 = split_text(lines, val_ratio=0.3, seed=7)
    assert train == train2 and val == val2

def test_batchify_text_order_and_size(tmp_path):
    p = tmp_path / "abc.txt"
    p.write_text("A\nB\nC\nD\n", encoding="utf-8")
    lines = load_text_dataset(str(p))
    batches = list(batchify_text(lines, batch_size=2, shuffle=False))
    assert batches == [["A", "B"], ["C", "D"]]
