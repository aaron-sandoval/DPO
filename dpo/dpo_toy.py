from transformers import GPT2Tokenizer
from rich import print as rprint
from rich.table import Table

from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    OnTheFlyDataLoader,
    DPOTrainingArgs,
)

# %%
LOW_GPU_MEM = False
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
dpo_model: DPOModel = DPOModel(model=BASE_MODEL)
ref_model: DPOModel = DPOModel(model=BASE_MODEL, fp16=True)
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
    judge_fn=selected_judge_fn, 
    implicit_reward_fn=selected_implicit_reward_fn,
    exp_name=judge_name,
)
# %%

# Create the on-the-fly dataset and dataloader
on_the_fly_dataset = OnTheFlyBinaryPreferenceDataset(
    prompt=args.prefix, 
    judge_fn=args.judge_fn, 
    implicit_reward_fn=args.implicit_reward_fn,
    gen_model=dpo_model, 
    num_samples=args.train_length
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
