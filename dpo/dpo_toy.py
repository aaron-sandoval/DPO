# %%
import torch as t
from transformers import GPT2Tokenizer
from rich import print as rprint
from rich.table import Table
from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    DPOTrainingArgs,
    DPOIMDBTrainer,
    judge_periods,
    reward_char_count,
)

# %%
LOW_GPU_MEM = False
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
dpo_model: DPOModel = DPOModel(model=BASE_MODEL, tokenizer=tokenizer)
ref_model: DPOModel = DPOModel(model=BASE_MODEL, tokenizer=tokenizer, fp16=True)
# %%

sample_ids, samples = dpo_model.generate(
    prompt="So long, and thanks for all the",
    batch_size=5,
    gen_len=20,
    temperature=0.8,
)

table = Table("Token IDs", "Samples", title="Demo of `sample` function", show_lines=True)

for ids, sample in zip(sample_ids, samples):
    table.add_row(str(ids.tolist()), repr(sample))

rprint(table)
# %%

selected_judge_fn = judge_periods
selected_implicit_reward_fn = reward_char_count

judge_name: str = selected_implicit_reward_fn.__name__

args = DPOTrainingArgs(
    base_model=BASE_MODEL,
    tokenizer=tokenizer,
    judge_fn=selected_judge_fn, 
    implicit_reward_fn=selected_implicit_reward_fn,
    exp_name=judge_name,
)
args.base_learning_rate = 4e-6
args.warmup_steps = 50
args.dpo_beta = 0.2
args.use_wandb = True
# %%

# Create the on-the-fly dataset and dataloader
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
trainer = DPOIMDBTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, args=args, ref_model=ref_model)
# %%
trainer.train()
# %%
