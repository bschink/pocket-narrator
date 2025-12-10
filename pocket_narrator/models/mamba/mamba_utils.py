# pocket_narrator/models/mamba/mamba_utils.py

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Setzt Zufallssamen für Python, NumPy und PyTorch
    für reproduzierbare Experimente.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Für deterministischere Ergebnisse (kann Training etwas verlangsamen)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# HF Dataset Wrapper
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    """
    Wrappt ein HuggingFace Dataset (oder Dataset-Split),
    so dass es mit einem PyTorch DataLoader benutzt werden kann.

    Erwartet pro Beispiel mindestens 'input_ids'.
    Optional: 'labels'. Falls 'labels' nicht existiert,
    werden sie identisch zu 'input_ids' gesetzt.
    """

    def __init__(
        self,
        hf_dataset: Any,
        input_key: str = "input_ids",
        label_key: str = "labels",
    ) -> None:
        """
        Args:
            hf_dataset: Ein HuggingFace Dataset-Objekt oder ähnlicher Mapping-Typ.
            input_key: Name der Spalte für Eingabesequenzen.
            label_key: Name der Spalte für Zielsequenzen (Labels).
        """
        self.ds = hf_dataset
        self.input_key = input_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.ds[idx]

        # input_ids holen und in Tensor umwandeln
        input_ids = example[self.input_key]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # labels holen oder aus input_ids ableiten
        if self.label_key in example and example[self.label_key] is not None:
            labels = example[self.label_key]
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
        else:
            # Fallback: Labels = Input (kein Shift) – reicht als Default
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
