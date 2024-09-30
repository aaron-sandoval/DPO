# %%
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Union

import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
# from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import pipeline, set_seed
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
MAIN = __name__ == "__main__"

LOW_GPU_MEM = False
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"

# %%
raw_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

class LangModel(nn.Module):
    def __init__(self, model: GPT2LMHeadModel=raw_model, tokenizer: GPT2Tokenizer=tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 30, num_return_sequences: int = 1) -> str:
        encoded = self.tokenizer(prompt, return_tensors='pt').to(device)
        completions: CausalLMOutputWithCrossAttentions = self.model(**encoded)
        temp0 = completions.logits.argmax(dim=-1)
        return self.tokenizer.decode(temp0, skip_special_tokens=True)

model = LangModel()
# generator = pipeline('text-generation', model=BASE_MODEL, device=device)
# set_seed(42)
# def generate_response(prompt: str, max_new_tokens: int = 30, num_return_sequences: int = 1) -> str:
#     completions = generator(prompt, max_length=max_new_tokens, num_return_sequences=num_return_sequences)
#     return [completion['generated_text'] for completion in completions]

# %%
if MAIN:
    print(model.generate("Hello, I'm a language model,"))

# %%
