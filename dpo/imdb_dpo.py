# %%
import random
from pathlib import Path

import torch as t
from transformers import GPT2Tokenizer

from utils import DATA_DIR, device, SEED
from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    DPOTrainingArgs,
    DPOTrainer,
    judge_periods, 
    reward_char_count,
)
from sentiment_judge import make_preference_pairs, judge_sentiment

# %%
BASE_MODEL = "gpt2-large"
sft_model_path = DATA_DIR / "models" / "sft" / "2024-10-03T1630_4000.pt"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

dpo_model: DPOModel = DPOModel(model=sft_model_path)
ref_model: DPOModel = DPOModel(model=sft_model_path, fp16=True)
# %%
# Baseline SFT generations and scoring
prefixes = [
    "The movie was ",
    "This is a story ",
    "I watched ",
    "People told me ",
    "I read ",
    "I heard ",
    "This thing about this movie was ",
    "After watching this movie, ",
    "This movie ",
    "Everyone says ",
]
count_positive, count_negative = 0, 0
num_generations = 64
batch_size = 64
gen_len = 50

def generate_and_judge(
        model: DPOModel, 
        prefixes: list[str], 
        num_generations: int = num_generations,
        batch_size: int = batch_size, 
        gen_len: int = gen_len,
    ) -> tuple[int, int]:
    assert num_generations % batch_size == 0
    count_positive, count_negative = 0, 0
    for _ in range(num_generations // batch_size):
        _, generations = model.generate(
            random.sample(prefixes, 1)[0],
            batch_size=batch_size,
            gen_len=gen_len,
        )
        judgments = judge_sentiment(generations)
        count_positive += t.sum(judgments == 1).item()
        count_negative += t.sum(judgments == -1).item()
    return count_positive, count_negative

sft_count_positive, sft_count_negative = generate_and_judge(dpo_model, prefixes, num_generations, batch_size, gen_len, count_positive, count_negative)
print(f"Positive judgments: {sft_count_positive}/{num_generations}")
print(f"Negative judgments: {sft_count_negative}/{num_generations}")
print(f"Positive proportion: {sft_count_positive / num_generations}")

# %%
args = DPOTrainingArgs(
    base_model=BASE_MODEL,
    tokenizer=tokenizer,
    judge_fn=judge_periods, 
    train_length=20,
    implicit_reward_fn=reward_char_count,
    exp_name="Single generation",
    prefixes=prefixes,
)
args.base_learning_rate = 4e-6
args.warmup_steps = 50
args.dpo_beta = 0.2
args.use_wandb = True
# %%
on_the_fly_dataset = OnTheFlyBinaryPreferenceDataset(
    prompt=args.prefix, 
    args=args,
    gen_model=dpo_model,
)
on_the_fly_dataloader = t.utils.data.DataLoader(
    on_the_fly_dataset,
    batch_size=args.batch_size,
)
# a = on_the_fly_dataset.generate_preference_pairs(batch_size=4)
a = next(iter(on_the_fly_dataloader))
assert isinstance(a, dict)
assert "preferred" in a and "rejected" in a and "prefix_len" in a
assert len(a["preferred"]) == len(a["rejected"]) == len(a["prefix_len"]) == args.batch_size
assert t.all(a["prefix_len"] == t.full((args.batch_size,), on_the_fly_dataset.prefix_len))
# %%
trainer = DPOTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, args=args, ref_model=ref_model)
# %%
trainer.train()
# %%