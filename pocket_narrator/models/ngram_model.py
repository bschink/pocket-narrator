"""
Contains the implementation of a simple n-gram baseline model.
"""
import os
import json
import random
from collections import defaultdict, Counter
from tqdm import tqdm
from .base_model import AbstractLanguageModel

class NGramModel(AbstractLanguageModel):
    def __init__(self, vocab_size: int, n: int, eos_token_id: int):
        """
        Initialize the n-gram model

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
            n (int): The order of the n-gram model (e.g., 3 for a trigram model).
        """
        super().__init__(vocab_size)
        if n < 2:
            raise ValueError("n must be at least 2 for n-gram models.")
        self.n = n
        self.eos_token_id = eos_token_id
        # n-gram counts: {('the', 'cat'): {'sat': 10, 'slept': 5}}
        self.ngram_counts = defaultdict(Counter)
        # context counts: {('the', 'cat'): 15}
        self.context_counts = defaultdict(int)

    def train(self, train_tokens: list[list[int]]):
        """
        Trains the n-gram model by counting sequence occurrences in the corpus.
        """
        print(f"INFO: Training {self.n}-gram model...")
        total_sequences = len(train_tokens)
        for sequence in tqdm(train_tokens, desc="Building n-gram counts", unit="sequence"):
            if len(sequence) < self.n:
                continue
            for i in range(len(sequence) - self.n + 1):
                # first n-1 tokens of the n-gram
                context = tuple(sequence[i : i + self.n - 1])
                # nth token
                target = sequence[i + self.n - 1]
                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1

    def _predict_next_token(self, context: tuple) -> int:
        """Predicts the most likely next token given a context."""
        if context in self.ngram_counts:
            most_common = self.ngram_counts[context].most_common(1)
            return most_common[0][0]
        else:
            # return random token if context unseen
            return random.randint(0, self.vocab_size - 1)
        
    def _would_create_repeated_ngram(self, history: list[int], candidate: int, ngram_size: int) -> bool:
        """
        Check if adding `candidate` to `history` would create an n-gram
        that already appears in the history.
        """
        if ngram_size is None or ngram_size < 2:
            return False
        if len(history) + 1 < ngram_size:
            return False

        # last ngram_size-1 tokens + candidate
        new_ngram = tuple(history[-(ngram_size - 1):] + [candidate])
        # slide over history
        for i in range(len(history) - ngram_size + 1):
            if tuple(history[i : i + ngram_size]) == new_ngram:
                return True
        return False

    def _choose_next_token(
        self,
        context: tuple,
        generated_sequence: list[int],
        strategy: str,
        no_repeat_ngram_size: int
    ) -> int:
        """
        Choose the next token according to the given strategy:
        - 'greedy': most frequent continuation (current default)
        - 'sample': sample from the distribution of continuations
        Optionally avoid repeating n-grams of size `no_repeat_ngram_size`.
        """
        # unseen context fallback
        if context not in self.ngram_counts:
            return random.randint(0, self.vocab_size - 1)

        items = list(self.ngram_counts[context].items())  # (token, count)

        # optional no-repeat n-gram filtering
        if no_repeat_ngram_size is not None:
            filtered = []
            for token, count in items:
                if not self._would_create_repeated_ngram(
                    generated_sequence,
                    token,
                    no_repeat_ngram_size,
                ):
                    filtered.append((token, count))
            if filtered:
                items = filtered

        if not items:
            # extremely rare fallback
            return random.randint(0, self.vocab_size - 1)

        # strategy selection
        strategy = strategy or "greedy"
        if strategy == "sample":
            tokens, counts = zip(*items)
            total = sum(counts)
            probs = [c / total for c in counts]
            return random.choices(tokens, weights=probs, k=1)[0]
        else:
            # greedy (original behavior)
            return max(items, key=lambda x: x[1])[0]
        
    def predict_sequence_batch(
        self,
        input_tokens_batch: list[list[int]],
        max_length: int = 400,
        strategy: str = "greedy",
        no_repeat_ngram_size: int = None,
    ) -> list[list[int]]:
        """
        Generates a sequence continuation for each prompt in the batch.
        - strategy: 'greedy' or 'sample'
        - no_repeat_ngram_size: if set (e.g., 3), try to avoid repeating
          n-grams of that size in the generated sequence.
        Stops generating if max_length is reached or an <eos> token is produced.
        """
        predictions = []
        for prompt_tokens in input_tokens_batch:
            generated_sequence = list(prompt_tokens)
            while len(generated_sequence) < len(prompt_tokens) + max_length:
                # last n-1 tokens of the current sequence
                context = tuple(generated_sequence[-(self.n - 1):])
                next_token = self._choose_next_token(
                    context,
                    generated_sequence,
                    strategy=strategy,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )

                if next_token == self.eos_token_id:
                    break

                generated_sequence.append(next_token)
            
            # return generated part only
            predictions.append(generated_sequence[len(prompt_tokens):])
        return predictions


    def save(self, model_path: str):
        """Saves the n-gram model's counts to a JSON file."""
        print(f"INFO: Saving n-gram model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # convert tuple keys to strings for JSON serialization
        serializable_counts = {str(k): v for k, v in self.ngram_counts.items()}
        
        data_to_save = {
            "config": {
                "model_type": "ngram",
                "vocab_size": self.vocab_size,
                "n": self.n,
                "eos_token_id": self.eos_token_id
            },
            "counts": serializable_counts
        }
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)

    @classmethod
    def load(cls, model_path: str, config: dict):
        """Loads the n-gram model's counts from a file."""
        print("INFO: Instantiating n-gram model from config...")
        n = int(config['n'])
        vocab_size = int(config['vocab_size'])
        eos_token_id = int(config['eos_token_id'])
        model = cls(vocab_size=vocab_size, n=n, eos_token_id=eos_token_id)

        with open(model_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # deserialize the string keys back into tuples and convert Counter keys to integers
        model.ngram_counts = defaultdict(
            Counter, 
            {eval(k): Counter({int(token): count for token, count in v.items()}) 
             for k, v in saved_data["counts"].items()}
        )
        
        for context, counter in model.ngram_counts.items():
            model.context_counts[context] = sum(counter.values())
            
        return model