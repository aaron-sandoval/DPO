from typing import Sequence

import torch as t
from transformers import pipeline
from jaxtyping import Bool

from utils import device

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    device=device,
)

def judge_sentiment(text: str | Sequence[str], device: str = "cpu") -> bool | Bool[t.Tensor, "N"]:
    if isinstance(text, str):
        return sentiment_analysis(text)[0]["label"] == "POSITIVE"
    else:
        return t.tensor(
            [judgment["label"] == "POSITIVE" for judgment in sentiment_analysis(text)],
            dtype=t.bool,
            device=device,
        )


print(judge_sentiment("I love this!"))
print(judge_sentiment(["I love this!", "It was OK, not sure what to think"]))
