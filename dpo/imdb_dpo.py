# %%
import random
from pathlib import Path
import pickle
from datetime import datetime
from typing import Optional

import wandb
import torch as t
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm

from utils import DATA_DIR, device, SEED
from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    OnTheFlySentimentPairDataset,
    DPOTrainingArgs,
    DPOTrainer,
    judge_periods, 
    reward_char_count,
    get_correct_token_logprobs,
    get_optimizer_and_scheduler,
)
from sentiment_judge import make_preference_pairs, judge_sentiment

# %%
BASE_MODEL = "gpt2-large"
sft_model_path = "xtremekiwi/gpt2-large_sft-imdb4k"
LOAD_PAIRS_DATASET = DATA_DIR / "pair_data" / "2024-10-04T1758.pkl"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

dpo_model: DPOModel = DPOModel(model=sft_model_path, tokenizer=tokenizer)
ref_model: DPOModel = DPOModel(model=sft_model_path, tokenizer=tokenizer, fp16=True)
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
num_generations = 64*100
batch_size = 64
gen_len = 100

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
        count_positive += t.sum(judgments > 0).item()
        count_negative += t.sum(judgments <= 0).item()
    return count_positive, count_negative

# %%
# sft_count_positive, sft_count_negative = generate_and_judge(dpo_model, prefixes, num_generations, batch_size, gen_len)
# print(f"Positive judgments: {sft_count_positive}/{num_generations}")
# print(f"Negative judgments: {sft_count_negative}/{num_generations}")
# print(f"Positive proportion: {sft_count_positive / num_generations}")

# %%
args = DPOTrainingArgs(
    base_model=BASE_MODEL,
    tokenizer=tokenizer,
    train_length=batch_size*10,
    # gen_len = 100,
    batch_size=batch_size,
    judge_fn=judge_periods, 
    implicit_reward_fn=reward_char_count,
    exp_name="IMDB DPO",
    prefixes=prefixes,
)
args.base_learning_rate = 4e-6
args.warmup_steps = 50
args.dpo_beta = 0.2
# %%
if LOAD_PAIRS_DATASET:
    with open(LOAD_PAIRS_DATASET, "rb") as f:
        pairs_data = pickle.load(f)
    on_the_fly_dataset = OnTheFlySentimentPairDataset(
        prefixes=prefixes, 
        args=args,
        gen_model=ref_model,
        dataset=pairs_data
    )
else:
    on_the_fly_dataset = OnTheFlySentimentPairDataset(
        prefixes=prefixes, 
        args=args,
        gen_model=ref_model,
    )
    fname = DATA_DIR / "pair_data" / f"{datetime.now().isoformat(timespec='minutes').replace(':', '')}.pkl"
    with open(fname, "wb") as f:
        f.write(pickle.dumps(on_the_fly_dataset.dataset))
on_the_fly_dataloader = t.utils.data.DataLoader(
    on_the_fly_dataset,
    batch_size=args.batch_size,
)
# a = on_the_fly_dataset.generate_preference_pairs(batch_size=4)
a = next(iter(on_the_fly_dataloader))
# assert isinstance(a, tuple)
# assert isinstance(a[0], str) and isinstance(a[1], str)
assert "preferred" in a and "rejected" in a and "prefix_len" in a
assert len(a["preferred"]) == len(a["rejected"]) == len(a["prefix_len"]) == args.batch_size
assert t.all(a["prefix_len"] == t.full((args.batch_size,), on_the_fly_dataset.prefix_len))
# %%
class DPOTrainer:
    def __init__(
            self, 
            model: DPOModel, 
            dataloader: t.utils.data.DataLoader, 
            args: DPOTrainingArgs,
            ref_model: Optional[DPOModel] = None,
            save_model: bool = True,
        ):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        self.save_model = save_model
        if ref_model is None:
            self.ref_model = DPOModel(model=args.base_model, fp16=True)
        else:
            self.ref_model = ref_model
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.step = 0
        self.save_model = save_model

    def dpo_loss(
            self, 
            logprobs_pref_trained: Float[Tensor, "batch gen_len"], 
            logprobs_rej_trained: Float[Tensor, "batch gen_len"], 
            logprobs_pref_ref: Float[Tensor, "batch gen_len"], 
            logprobs_rej_ref: Float[Tensor, "batch gen_len"],
        ) -> Tensor:
        """
        Computes the DPO loss for a batch of logits and rewards.
        """
        sigmoid_arg = self.args.dpo_beta * (logprobs_pref_trained - logprobs_pref_ref - (logprobs_rej_ref - logprobs_rej_trained))
        loss = -nn.functional.logsigmoid(sigmoid_arg).mean()
        if self.args.use_wandb:
            wandb.log({
                "loss": loss,
                "logprobs_pref_trained": logprobs_pref_trained,
                "logprobs_rej_trained": logprobs_rej_trained,
                "logprobs_pref_ref": logprobs_pref_ref,
                "logprobs_rej_ref": logprobs_rej_ref,
            }, step=self.step)
        return loss
    
    def _train(self):
        for batch in tqdm(self.dataloader):
            if self.args.use_wandb:
                # _, gen_strings = self.model.generate(
                #     self.dataloader.dataset.prompt, 
                #     batch_size=self.args.batch_size, 
                #     gen_len=self.args.gen_len,
                #     temperature=self.args.temperature,
                # )
                pref_strs: list[str] = [self.args.tokenizer.decode(ids) for ids in batch["preferred"]]
                rej_strs: list[str] = [self.args.tokenizer.decode(ids) for ids in batch["rejected"]]
                # gen_rewards = t.tensor([self.implicit_reward_fn(gen_str) for gen_str in pref_strs+rej_strs], dtype=t.float16, requires_grad=False)
                avg_pref_reward: float = sum(self.implicit_reward_fn(pref_str) for pref_str in pref_strs) / len(pref_strs)
                avg_rej_reward: float = sum(self.implicit_reward_fn(rej_str) for rej_str in rej_strs) / len(rej_strs)
                print(pref_strs[0])
                wandb.log({
                    "reward": {
                        "preferred": avg_pref_reward,
                        "rejected": avg_rej_reward,
                        # "mean": gen_rewards.mean().item(),
                        # "all": gen_rewards,
                    },
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=self.step)
            self.optimizer.zero_grad()
            preferred_ids = batch["preferred"].to(device)
            rejected_ids = batch["rejected"].to(device)
            prefix_lens = batch["prefix_len"]
            # assert t.all(batch["prefix_len"] == prefix_lens)
            preferred_logits = self.model(preferred_ids)
            rejected_logits = self.model(rejected_ids)
            preferred_logprobs = get_correct_token_logprobs(preferred_logits, preferred_ids, prefix_len=prefix_lens)
            rejected_logprobs = get_correct_token_logprobs(rejected_logits, rejected_ids, prefix_len=prefix_lens)
            with t.inference_mode():
                preferred_ref_logits = self.ref_model(preferred_ids)
                rejected_ref_logits = self.ref_model(rejected_ids)
            preferred_ref_logprobs = get_correct_token_logprobs(preferred_ref_logits, preferred_ids, prefix_len=prefix_lens)
            rejected_ref_logprobs = get_correct_token_logprobs(rejected_ref_logits, rejected_ids, prefix_len=prefix_lens)
            loss = self.dpo_loss(preferred_logprobs, rejected_logprobs, preferred_ref_logprobs, rejected_ref_logprobs)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

    def train(self):
        self.model.train()
        self.ref_model.eval()
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=f"{self.args.exp_name}_seed={self.args.seed}_{datetime.now().isoformat(timespec='minutes').replace(':', '')}",
                config=self.args,
            )
            try:
                self._train()
            finally:
                if self.save_model:
                    self.model.save_model()
                wandb.finish()
        else:
            self._train()
            if self.save_model:
                self.model.save_model()

# %%
trainer = DPOTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, args=args, ref_model=ref_model)
# %%
args.use_wandb = False
trainer.train()
# %%