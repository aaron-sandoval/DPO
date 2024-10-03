# %%
from pathlib import Path

import torch as t
from transformers import GPT2Tokenizer

from utils import DATA_DIR, device, SEED
from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    OnTheFlyDataLoader,
    DPOTrainingArgs,
    DPOTrainer,
)

# %%
BASE_MODEL = "gpt2-large"
sft_model_path = DATA_DIR / "models" / "sft" / "2024-10-03T1630_4000.pt"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

dpo_model: DPOModel = DPOModel(model=sft_model_path)
ref_model: DPOModel = DPOModel(model=sft_model_path, fp16=True)
# %%
