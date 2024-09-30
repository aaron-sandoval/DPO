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
d_model: int = raw_model.config.n_embd
d_vocab: int = len(tokenizer)
# %%

class DPOModel(nn.Module):
    def __init__(self, model: GPT2LMHeadModel=raw_model, tokenizer: GPT2Tokenizer=tokenizer):
        super().__init__()
        self.model = model
        self.base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
        self.tokenizer = tokenizer

    def generate(self, prompt: str, batch_size: int = 1, temperature: float = 1.0) -> str:
        encoded = self.tokenizer(prompt, return_tensors='pt').to(device)
        completion = self.model.generate(encoded.input_ids) #, max_new_tokens=100, temperature=temperature, num_return_sequences=batch_size)
        # completions: CausalLMOutputWithCrossAttentions = self.model(**encoded)
        # probs = (completions.logits[0]/temperature).softmax(dim=-1)
        # sampled_tokens = t.multinomial(probs, num_samples=batch_size, replacement=True)
        # # output_ids = t.cat([encoded.input_ids[0], sampled_tokens.squeeze()])
        return self.tokenizer.decode(completion.squeeze(), skip_special_tokens=True)

dpo_model: DPOModel = DPOModel()
# %%
if MAIN:
    print(dpo_model.generate("What is the capital of France?"))

# %%
