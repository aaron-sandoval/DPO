from typing import Sequence, Generator
from itertools import combinations

import torch as t
from transformers import pipeline
from jaxtyping import Int, Float

from utils import device

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    device=device,
)

def judge_sentiment(text: str | Sequence[str], device: str = "cpu") -> int | Int[t.Tensor, "N"]:
    if isinstance(text, str):
        return 1 if sentiment_analysis(text)[0]["label"] == "POSITIVE" else -1
    else:
        return t.tensor(
            [(1 if judgment["label"] == "POSITIVE" else -1)*judgment["score"] for judgment in sentiment_analysis(text)],
            dtype=t.bool,
            device=device,
        )


def sort_by_sentiment(texts: Sequence[str], device: str = "cpu") -> list[str]:
    """
    Sorts the given texts by sentiment score in ascending order.
    """
    judgments: Float[t.Tensor, "N"] = judge_sentiment(texts, device=device)
    return [text for _, text in sorted(zip(judgments, texts), key=lambda x: x[0])]


def make_preference_pairs(
    texts: Sequence[str],
    device: str = "cpu",
) -> Generator[tuple[str, str], None, None]:
    """Returns a generator of all possible preference pairs, positive first."""
    ranked_texts = sort_by_sentiment(texts, device=device)
    return combinations(ranked_texts, 2)


if __name__ == "__main__":
    # print(judge_sentiment("I love this!"))
    # print(judge_sentiment(["I love this!", "It was OK, not sure what to think"]))
    print(list(make_preference_pairs(["I love this!", "It was a mixed experience", "I hated it"])))
