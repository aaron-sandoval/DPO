# %%
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Union, Sequence, Any, Literal, Optional

import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
# from eindex import eindex
from jaxtyping import Float, Int, Bool
from rich import print as rprint
from rich.table import Table
from torch import Tensor
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer, logging
# from transformers import pipeline, set_seed
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


logging.set_verbosity_warning()
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
MAIN = __name__ == "__main__"

LOW_GPU_MEM = False
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"

# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

# %%

class DPOModel(nn.Module):
    def __init__(
            self, 
            model: GPT2LMHeadModel=GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device), 
            tokenizer: GPT2Tokenizer=tokenizer
        ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.d_model: int = self.model.config.n_embd
        self.d_vocab: int = len(self.tokenizer)

    def forward(self, input_ids: Int[Tensor, "batch seq_len"], **kwargs):
        return self.model(input_ids=input_ids, **kwargs).logits

    def generate(
            self,
            prompt: str,
            batch_size: int = 1,
            gen_len: int = 20,
            temperature: float = 1.0,
            **kwargs
        ) -> tuple[Int[Tensor, "batch_size prompt_len_plus_gen_len"], list[str]]:
        encoded = self.tokenizer(prompt, return_tensors='pt')
        completion = self.model.generate(
            encoded.input_ids.to(self.model.device),
            max_new_tokens=gen_len,
            temperature=temperature,
            num_return_sequences=batch_size,
            do_sample=True,
            **kwargs
        )
        return completion, [self.tokenizer.decode(c, skip_special_tokens=True) for c in completion]

dpo_model: DPOModel = DPOModel()
ref_model: DPOModel = DPOModel()
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
# Reward and judge functions: simulating human preferences
def reward_char_count(sample: str, char: str = '.', *args, **kwargs) -> int:
    return sample.count(char)

def reward_to_judge(reward_fn: Callable[[str], float | int], *args, **kwargs) -> Callable[[Sequence[str], Sequence[str]], Bool[Tensor, "batch"]]:
    """
    Converts a reward function to a judge function.
    """
    def judge_fn(samples0: Sequence[str], samples1: Sequence[str], tokenizer: PreTrainedTokenizer=tokenizer) -> Int[Tensor, "batch"]:
        rewards0 = t.tensor([reward_fn(s, *args, tokenizer=tokenizer, **kwargs) for s in samples0], requires_grad=False)
        rewards1 = t.tensor([reward_fn(s, *args, tokenizer=tokenizer, **kwargs) for s in samples1], requires_grad=False)
        return rewards0 > rewards1
    return judge_fn

judge_periods = reward_to_judge(reward_char_count, char='.')

assert t.all(judge_periods(["This is a test.", "This is a test.", "This is a test."], ["This is a test", "This is a test..", "This. is a test."]) == t.tensor([True, False, False]))
# %%

@dataclass
class DPOTrainingArgs():
    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()

    # Wandb / logging
    exp_name: str = "DPO_Implementation"
    wandb_project_name: str | None = "capstone_dpo"
    wandb_entity: str | None = None  
    use_wandb: bool = False

    # Duration of different phases
    train_length: int = 64*200
    batch_size: int = 64
    # num_minibatches: int = 4
    # batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    base_learning_rate: float = 1e-6  # Rafailov et al. 2024
    # head_learning_rate: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 150  # Rafailov et al. 2024
    final_scale: float = 0.1

    # Computing other PPO loss functions
    # clip_coef: float = 0.2
    # vf_coef: float = 0.15
    # ent_coef: float = 0.001

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 0.6
    prefix: str = "This is"

    # Extra stuff for RLHF
    dpo_beta: float = 0.1
    reward_fn: Callable = judge_periods
    normalize_reward: bool = False

# %%
def get_optimizer(args: DPOTrainingArgs, model: DPOModel) -> t.optim.Optimizer:
    """
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    """
    return t.optim.Adam(params=model.model.parameters(), lr = args.base_learning_rate)


args = DPOTrainingArgs()
optimizer = get_optimizer(args, dpo_model)

# %%
def get_lr_scheduler(warmup_steps, total_steps, final_scale):
    """
    Creates an LR scheduler that linearly warms up for `warmup_steps` steps,
    and then linearly decays to `final_scale` over the remaining steps.
    """
    def lr_lambda(step):
        assert step <= total_steps, f"Step = {step} should be less than total_steps = {total_steps}."
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1 - (1 - final_scale) * (step - warmup_steps) / (total_steps - warmup_steps)

    return lr_lambda


def get_optimizer_and_scheduler(args: DPOTrainingArgs, model: DPOModel):
    optimizer = get_optimizer(args, model)
    lr_lambda = get_lr_scheduler(args.warmup_steps, args.train_length//args.batch_size, args.final_scale)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

# %%
def get_correct_token_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int | None = None,
) -> Float[Tensor, "batch gen_len"]:
    """
    Returns correct logprobs for the given logits and tokens, for all the tokens
    after the prefix tokens (which have length equal to `prefix_len`).

    If prefix_len = None then we return shape (batch, seq_len-1). If not, then
    we return shape (batch, seq_len-prefix_len) representing the predictions for
    all tokens after the prefix tokens.
    """
    # Using no prefix_len argument is equivalent to prefix_len=1
    prefix_len = prefix_len or 1

    # Slice logprobs and tokens, so that each logprob matches up with the token which it predicts
    logprobs = logits[:, prefix_len - 1 : -1].log_softmax(-1)
    correct_tokens = tokens[:, prefix_len:]

    # correct_logprobs[batch, seq] = logprobs[batch, seq, correct_tokens[batch, seq]]
    # correct_logprobs = eindex(logprobs, correct_tokens, "batch seq [batch seq] -> batch seq")
    correct_logprobs = t.gather(logprobs, -1, correct_tokens.unsqueeze(-1)).squeeze(-1)

    assert correct_logprobs.shape == (tokens.shape[0], tokens.shape[1] - prefix_len)
    return correct_logprobs

# %%
# HF Dataset
# hf_dataset_name = "Anthropic/hh-rlhf"
# preferred_column: str = "chosen"
# rejected_column: str = "rejected"
# def collate_prompt_integrated(
#         batch: Sequence[dict[str, str]], 
#         tokenizer: PreTrainedTokenizer, 
#         max_length: int = args.gen_len, 
#         device: t.device = device
#     ):
#     """Collate function for dataset where the prompt is already concatenated with the text completions.
#     """
#     preferred_tokens: list[Int[Tensor, "seq_len"]] = [tokenizer(item[preferred_column], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device) for item in batch]
#     rejected_tokens: list[Int[Tensor, "seq_len"]] = [tokenizer(item[rejected_column], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device) for item in batch]
    

# hf_dataset = datasets.load_dataset(hf_dataset_name, split="train")
# train_dataloader = t.utils.data.DataLoader(hf_dataset, batch_size=args.batch_size, collate_fn=collate_prompt_integrated, shuffle=True)

    
# %%
# On the fly dataloader

class OnTheFlyBinaryPreferenceDataset(t.utils.data.Dataset):
    def __init__(
            self, 
            prompt: str, 
            judge_fn: Callable[[Sequence[str], Sequence[str]], Bool[Tensor, "batch"]],
            gen_model: DPOModel = ref_model, 
            num_samples: int = args.train_length,
        ):
        """
        Args:
            prompt: The prompt to use for generating the completions.
            judge_fn: A function that takes in pairs of generated completions and returns a tensor indexing the preferred completion.
            ref_model: The reference model to use for generating the completions.
            tokenizer: The tokenizer to use for encoding the completions.
            num_samples: The number of samples to generate.
        """
        self.prompt = prompt
        self.prefix_len = len(tokenizer(prompt)["input_ids"])
        self.judge_fn = judge_fn
        self.gen_model = gen_model
        self.num_samples = num_samples
        self.encoded_prompt = tokenizer(self.prompt, return_tensors="pt").to(device)
        self.cache_batch_size = args.batch_size
        self.cache: list[tuple[Int[Tensor, "seq_len"], Int[Tensor, "seq_len"]]] = [0] * self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache[idx] == 0:
            preferred, rejected = self.generate_preference_pairs(batch_size=self.cache_batch_size)
            cache_slice = slice(idx-idx%self.cache_batch_size, idx-idx%self.cache_batch_size+self.cache_batch_size)
            self.cache[cache_slice] = zip(preferred, rejected)
        return dict(zip(("preferred", "rejected", "prefix_len"), (*self.cache[idx], self.prefix_len)))
        
    @t.inference_mode()
    def generate_preference_pairs(
        self,
        batch_size: int,
        temperature: float = 1.0,
        gen_len: int = args.gen_len,
        device: t.device = device
    ) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
        """Generate a batch of preferred and rejected completions using the reference model."""
        gen_tokens, gen_strings = self.gen_model.generate(
            self.prompt,
            gen_len=gen_len,
            batch_size=batch_size*2,
            temperature=temperature,
        )
        preferred_index_0: Bool[Tensor, "batch"] = self.judge_fn(gen_strings[:batch_size], gen_strings[batch_size:], tokenizer=tokenizer)
        preferred_mask: Bool[Tensor, "2 batch"] = t.stack([preferred_index_0, ~preferred_index_0], dim=0).to(device)
        # Reshape gen_tokens into (2, batch_size, seq_len)
        gen_tokens_reshaped = einops.rearrange(gen_tokens, "(b1 b2) ... -> ... b1 b2", b1=2)
        # Gather preferred and rejected completions
        preferred_tokens = gen_tokens_reshaped.masked_select(preferred_mask)
        rejected_tokens = gen_tokens_reshaped.masked_select(~preferred_mask)
        preferred_tokens = einops.rearrange(preferred_tokens, "(seq batch) -> batch seq", batch=batch_size)
        rejected_tokens = einops.rearrange(rejected_tokens, "(seq batch) -> batch seq", batch=batch_size)
        return preferred_tokens, rejected_tokens

# Create the on-the-fly dataset and dataloader
on_the_fly_dataset = OnTheFlyBinaryPreferenceDataset(
    prompt=args.prefix, 
    judge_fn=judge_periods, 
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
class DPOTrainer:
    def __init__(
            self, 
            model: DPOModel, 
            dataloader: t.utils.data.DataLoader, 
            ref_model: Optional[DPOModel] = None,
            args: DPOTrainingArgs = args
        ):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        if ref_model is None:
            self.ref_model = DPOModel()
        else:
            self.ref_model = ref_model
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.step = 0

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
                "pref_relative_logprob": logprobs_pref_trained - logprobs_pref_ref,
                "rej_relative_logprob": logprobs_rej_trained - logprobs_rej_ref,
            }, step=self.step)
        return loss
    
    def train(self):
        wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity,
            name=f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}",
            config=self.args,
        )
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            preferred_ids = batch["preferred"].to(device)
            rejected_ids = batch["rejected"].to(device)
            prefix_len = batch["prefix_len"][0].item()
            assert t.all(batch["prefix_len"] == prefix_len)
            preferred_logits = self.model(preferred_ids)
            rejected_logits = self.model(rejected_ids)
            preferred_logprobs = get_correct_token_logprobs(preferred_logits, preferred_ids, prefix_len=prefix_len)
            rejected_logprobs = get_correct_token_logprobs(rejected_logits, rejected_ids, prefix_len=prefix_len)
            with t.inference_mode():
                preferred_ref_logits = self.ref_model(preferred_ids)
                rejected_ref_logits = self.ref_model(rejected_ids)
            preferred_ref_logprobs = get_correct_token_logprobs(preferred_ref_logits, preferred_ids, prefix_len=prefix_len)
            rejected_ref_logprobs = get_correct_token_logprobs(rejected_ref_logits, rejected_ids, prefix_len=prefix_len)
            loss = self.dpo_loss(preferred_logprobs, rejected_logprobs, preferred_ref_logprobs, rejected_ref_logprobs)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

# %%
args.use_wandb = True
trainer = DPOTrainer(dpo_model, on_the_fly_dataloader)
trainer.train()
# %%
if MAIN:
    print(dpo_model.generate("What is the capital of France?", batch_size=3))

# %%
