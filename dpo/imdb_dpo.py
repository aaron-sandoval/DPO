# %%
import random
from pathlib import Path
import pickle
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import wandb
import torch as t
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm

from utils import DATA_DIR, device, SEED, LOW_GPU_MEM
from dpo import (
    DPOModel,
    OnTheFlyBinaryPreferenceDataset,
    OnTheFlySentimentPairDataset,
    DPOTrainingArgs,
    judge_periods, 
    reward_char_count,
    get_correct_token_logprobs,
    get_optimizer_and_scheduler,
    get_correct_token_logprobs_variable_prefix,
)
from sentiment_judge import make_preference_pairs, judge_sentiment

# %%
BASE_MODEL = "gpt2-large"
sft_model_path = "gpt2" if LOW_GPU_MEM else "xtremekiwi/gpt2-large_sft-imdb4k"
LOAD_PAIRS_DATASET = DATA_DIR / "pair_data" / "2024-10-04T1758.pkl"
# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

dpo_model: DPOModel = DPOModel(model=sft_model_path, tokenizer=tokenizer)
ref_model: DPOModel = DPOModel(model=sft_model_path, tokenizer=tokenizer, fp16=True)
# %%
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

@dataclass
class DPOIMDBTrainingArgs(DPOTrainingArgs):
    # eval_batches: int = 1
    train_batches_per_eval: int = 3  # Training batches before generating and evaluating metrics
    epochs: int = 1


args = DPOIMDBTrainingArgs(
    base_model=BASE_MODEL,
    tokenizer=tokenizer,
    train_length=64*10,
    # gen_len = 100,
    batch_size=64,
    exp_name="IMDB DPO",
    prefixes=prefixes,
    base_learning_rate=4e-6,
    warmup_steps=2,
    dpo_beta=0.2,
)
# %%
def evaluate_generation_sentiment(
        model: DPOModel, 
        prefixes: list[str] = prefixes, 
        num_batches: int = 5, 
        batch_size: int = 64, 
        gen_len: int = 100
    ) -> tuple[int, int]:
    """
    Evaluate the sentiment of generations from a model.
    """
    count_positive, count_negative = 0, 0
    for _ in range(num_batches):
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
# Baseline SFT generations and scoring
# sft_count_positive, sft_count_negative = generate_and_judge(dpo_model, prefixes, num_generations, batch_size, gen_len)
# print(f"Positive judgments: {sft_count_positive}/{num_generations}")
# print(f"Negative judgments: {sft_count_negative}/{num_generations}")
# print(f"Positive proportion: {sft_count_positive / num_generations}")

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
# assert t.all(a["prefix_len"] == t.full((args.batch_size,), on_the_fly_dataset.prefix_len))
# %%
class DPOIMDBTrainer:
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

    def evaluate_generation_sentiment(self):
        """
        Generate and judge a batch of generations from `self.model`.
        """
        _, generations = self.model.generate(
            self.args.prefixes[0], 
            batch_size=self.args.batch_size, 
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
        )
        judgments = judge_sentiment(generations)
        if self.args.use_wandb:
            wandb.log({
                "eval_sentiment": judgments,
                "eval_sentiment_mean": judgments.mean().item(),
            }, step=self.step)
        else:
            positive_count = t.sum(judgments > 0).item()
            negative_count = t.sum(judgments <= 0).item()
            # print(f"Positive judgments: {positive_count}/{self.args.eval_batches}")
            # print(f"Negative judgments: {negative_count}/{self.args.eval_batches}")
            print(f"Positive proportion: {positive_count / self.args.batch_size}")
            print(f"Eval sentiment mean: {judgments.mean().item()}")
        return judgments

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
        for epoch in range(self.args.epochs):
            for i, batch in tqdm(enumerate(self.dataloader), desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                if i % self.args.train_batches_per_eval == 0:
                    self.evaluate_generation_sentiment()
                if self.args.use_wandb:
                    wandb.log({
                        # "reward": {
                            # "preferred": avg_pref_reward,
                            # "rejected": avg_rej_reward,
                            # "mean": gen_rewards.mean().item(),
                            # "all": gen_rewards,
                        # },
                        "lr": self.scheduler.get_last_lr()[0],
                    }, step=self.step)
                self.optimizer.zero_grad()
                # Encode the preferred IDs
                preferred_encoded_ids = [t.tensor(self.args.tokenizer.encode(s)) for s in batch["preferred"]]

                # Pad the sequences
                preferred_padded_ids =  t.nn.utils.rnn.pad_sequence(preferred_encoded_ids, batch_first=True, padding_value=self.args.tokenizer.eos_token_id)
                # Move the padded sequences to the device
                preferred_ids = preferred_padded_ids.to(device)
                
                # Encode the rejected IDs
                rejected_encoded_ids = [t.tensor(self.args.tokenizer.encode(s)) for s in batch["rejected"]]
                rejected_padded_ids =  t.nn.utils.rnn.pad_sequence(rejected_encoded_ids, batch_first=True, padding_value=self.args.tokenizer.eos_token_id)
                rejected_ids = rejected_padded_ids.to(device)
                
                prefix_lens = batch["prefix_len"]
                # assert t.all(batch["prefix_len"] == prefix_lens)
                preferred_logits = self.model(preferred_ids)
                rejected_logits = self.model(rejected_ids)
                preferred_logprobs = get_correct_token_logprobs_variable_prefix(preferred_logits, preferred_ids, gen_len=self.args.gen_len, prefix_len=prefix_lens)
                rejected_logprobs = get_correct_token_logprobs_variable_prefix(rejected_logits, rejected_ids, gen_len=self.args.gen_len, prefix_len=prefix_lens)
                with t.inference_mode():
                    preferred_ref_logits = self.ref_model(preferred_ids)
                    rejected_ref_logits = self.ref_model(rejected_ids)
                preferred_ref_logprobs = get_correct_token_logprobs_variable_prefix(preferred_ref_logits, preferred_ids, gen_len=self.args.gen_len, prefix_len=prefix_lens)
                rejected_ref_logprobs = get_correct_token_logprobs_variable_prefix(rejected_ref_logits, rejected_ids, gen_len=self.args.gen_len, prefix_len=prefix_lens)
                loss = self.dpo_loss(preferred_logprobs, rejected_logprobs, preferred_ref_logprobs, rejected_ref_logprobs)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.step += 1
            self.evaluate_generation_sentiment()

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
trainer = DPOIMDBTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, args=args, ref_model=ref_model)
# %%
args.use_wandb = True
trainer.train()
# %%
# Save to disk and/or Hugging Face
# dpo_model.save_model()
dpo_model.model.push_to_hub("gpt2-large-imdb-positive")
# %%

