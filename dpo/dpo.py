# %%
import sys
import time
from datetime import datetime
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
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

LOW_GPU_MEM = False
BASE_MODEL = "gpt2" if LOW_GPU_MEM else "gpt2-medium"

# %%
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)

# %%

class DPOModel(nn.Module):
    def __init__(
            self, 
            model: str = BASE_MODEL, 
            tokenizer: GPT2Tokenizer=tokenizer,
            fp16: bool = False,
        ):
        super().__init__()
        if fp16:
            self.model = GPT2LMHeadModel.from_pretrained(model).half().to(device)
        else:
            
            self.model = GPT2LMHeadModel.from_pretrained(model).to(device)
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

    def save_model(self, path: Optional[str] = None, suffix: Optional[str] = None):
        if path is None:
            path = DATA_DIR / "models"
        if suffix is None:
            suffix = ""
        dt = datetime.now().isoformat(timespec='minutes').replace(':', '')
        path = path / f"{dt}{suffix}.pt"
        self.model.save_pretrained(path)

    @classmethod
    def load_model(cls, name: str, **kwargs):
        path = DATA_DIR / "models" / f"{name}.pt"
        return cls(model=path, **kwargs)


dpo_model: DPOModel = DPOModel()
ref_model: DPOModel = DPOModel(fp16=True)
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

def reward_up_to_max_length(sample: str, length: int = 150, *args, **kwargs) -> int:
    return len(sample) if len(sample) <= length else -1

def reward_target_length(sample: str, length: int = 80, *args, **kwargs) -> int:
    return - abs(length - len(sample))

def reward_vowel_proportion(sample: str, *args, **kwargs) -> float:
    return sum(c.lower() in "aeiou" for c in sample) / len(sample)

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
judge_up_to_max_length = reward_to_judge(reward_up_to_max_length)
judge_target_length = reward_to_judge(reward_target_length)
judge_vowel_proportion = reward_to_judge(reward_vowel_proportion)
assert t.all(judge_periods(["This is a test.", "This is a test.", "This is a test."], ["This is a test", "This is a test..", "This. is a test."]) == t.tensor([True, False, False]))
# %%

@dataclass
class DPOTrainingArgs():
    # Judge and reward functions
    judge_fn: Callable[[Sequence[str], Sequence[str]], Bool[Tensor, "batch"]]
    implicit_reward_fn: Optional[Callable[[str], float | int]] = None

    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()

    # Wandb / logging
    exp_name: str = "DPO"
    wandb_project_name: str | None = "capstone_dpo"
    wandb_entity: str | None = None  
    use_wandb: bool = False

    # Duration of different phases
    train_length: int = 64*600
    batch_size: int = 64

    # Optimization hyperparameters
    base_learning_rate: float = 1e-6  # Rafailov et al. 2024
    # head_learning_rate: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 150  # Rafailov et al. 2024
    final_scale: float = 0.1

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 0.6
    prefix: str = "This is"

    # Extra stuff for DPO
    dpo_beta: float = 0.1


args = DPOTrainingArgs(
    judge_fn=judge_periods, 
    implicit_reward_fn=reward_char_count,
)
# %%
def get_optimizer(args: DPOTrainingArgs, model: DPOModel) -> t.optim.Optimizer:
    """
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    """
    return t.optim.Adam(params=model.model.parameters(), lr = args.base_learning_rate)

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
            implicit_reward_fn: Optional[Callable[[str], float | int]] = None,
            gen_model: DPOModel = dpo_model, 
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
class DPOTrainer:
    def __init__(
            self, 
            model: DPOModel, 
            dataloader: t.utils.data.DataLoader, 
            ref_model: Optional[DPOModel] = None,
            save_model: bool = True,
            args: DPOTrainingArgs = args
        ):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        if ref_model is None:
            self.ref_model = DPOModel(fp16=True)
        else:
            self.ref_model = ref_model
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.step = 0
        self.save_model = save_model
        self.judge_fn_name = self.args.judge_fn.__name__
        if hasattr(self.dataloader.dataset, "implicit_reward_fn"):
            self.implicit_reward_fn = self.dataloader.dataset.implicit_reward_fn
        else:
            self.implicit_reward_fn = None

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
                # "pref_relative_logprob": logprobs_pref_trained - logprobs_pref_ref,
                # "rej_relative_logprob": logprobs_rej_trained - logprobs_rej_ref,
            }, step=self.step)
        return loss
    
    def _train(self):
        for batch in tqdm(self.dataloader):
            if self.args.use_wandb and self.implicit_reward_fn is not None:
                # _, gen_strings = self.model.generate(
                #     self.dataloader.dataset.prompt, 
                #     batch_size=self.args.batch_size, 
                #     gen_len=self.args.gen_len,
                #     temperature=self.args.temperature,
                # )
                pref_strs: list[str] = [tokenizer.decode(ids) for ids in batch["preferred"]]
                rej_strs: list[str] = [tokenizer.decode(ids) for ids in batch["rejected"]]
                gen_rewards = t.tensor([self.implicit_reward_fn(gen_str) for gen_str in pref_strs+rej_strs], dtype=t.float16, requires_grad=False)
                # pref_rewards: list[float] = [self.implicit_reward_fn(pref_str) for pref_str in pref_strs]
                # rej_rewards: list[float] = [self.implicit_reward_fn(rej_str) for rej_str in rej_strs]
                # all_rewards = t.tensor(pref_rewards + rej_rewards, requires_grad=False)
                # avg_pref_reward: float = sum(pref_rewards) / len(pref_strs)
                # avg_rej_reward: float = sum(rej_rewards) / len(rej_strs)
                print(pref_strs[0])
                wandb.log({
                    "reward": {
                        # "mean_preferred": avg_pref_reward,
                        # "mean_rejected": avg_rej_reward,
                        "mean": gen_rewards.mean().item(),
                        "all": gen_rewards ,
                    },
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=self.step)
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

    def train(self):
        self.model.train()
        self.ref_model.eval()
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=f"{self.args.exp_name}_seed={self.args.seed}_{datetime.now().isoformat(timespec='minutes').replace(':', '')}_{self.judge_fn_name}",
                config=self.args,
            )
            try:
                self._train()
            finally:
                if self.save_model:
                    self.model.save_model(suffix=f"_{self.judge_fn_name}_{self.step}")
                wandb.finish()
        else:
            self._train()
        

# %%
args.base_learning_rate = 3e-6
# args.final_scale = 0.2
# args.dpo_beta = 0.2
args.use_wandb = True
trainer = DPOTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, ref_model=ref_model)
trainer.train()
# %%
if MAIN:
    print(dpo_model.generate("What is the capital of France?", batch_size=3))

# %%
def print_model_layer_dtype(model):
    print('\nModel dtypes:')
    for name, param in model.named_parameters():
        print(f"Param: {name}\tdtype: {param.dtype}")
# %%
gen_tokens, gen_strings = dpo_model.generate(
            on_the_fly_dataset.prompt,
            gen_len=args.gen_len,
            batch_size=4,
            temperature=args.temperature,
        )
print(gen_strings)
# %%

