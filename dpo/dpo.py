# %%
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Union, Sequence, Any, Literal

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
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer
# from transformers import pipeline, set_seed
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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

    def forward(self, input_ids: Int[Tensor, "batch seq_len"], attention_mask: Int[Tensor, "batch seq_len"], **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(
            self,
            prompt: str,
            batch_size: int = 1,
            gen_len: int = 20,
            temperature: float = 1.0,
            **kwargs
        ) -> tuple[Int[Tensor, "batch_size prompt_len_plus_gen_len"], list[str]]:
        encoded = self.tokenizer(prompt, return_tensors='pt').to(device)
        completion = self.model.generate(
            encoded.input_ids,
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
def reward_fn_char_count(generated_sample: str | list[str], char: str = '.') -> float | Float[Tensor, "batch"]:
    """
    Reward function, evaluated on the generated samples.

    In this case it's very simple: it just counts the number of instances of a particular character in
    the generated sample. It returns a tensor of rewards of dtype float the input is a list, or a single
    reward (float) if the input is a string.
    """
    if isinstance( generated_sample, str):
        return float(generated_sample.count(char))
    else:
        return t.tensor([float(s.count(char)) for s in generated_sample]).to(device)
    
# Test your reward function
A = 'This is a test.'
B = '......'
C = 'Whatever'
assert isinstance(reward_fn_char_count(A), float)
assert reward_fn_char_count(A) == 1
assert reward_fn_char_count(B) == 6
assert reward_fn_char_count(C) == 0
assert reward_fn_char_count([A, B, C]).dtype == t.float
assert reward_fn_char_count([A, B, C]).tolist() == [1.0, 6.0, 0.0]

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
    train_length: int = 64*40
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
    kl_coef: float = 1.0
    reward_fn: Callable = reward_fn_char_count
    normalize_reward: bool = False

# %%
def get_optimizer(args: DPOTrainingArgs, model: DPOModel) -> t.optim.Optimizer:
    """
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    """
    return t.optim.Adam(params=model.model.parameters(), lr = args.base_learning_rate, maximize = True)


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
    lr_lambda = get_lr_scheduler(args.warmup_steps, args.total_phases, args.final_scale)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

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
# Reward and judge functions: simulating human preferences
def reward_char_count(generated_sample: str, char: str = '.') -> float:
    """
    Reward function, evaluated on the generated samples.

    In this case it's very simple: it just counts the number of instances of a particular character in
    the generated sample. It returns a tensor of rewards of dtype float the input is a list, or a single
    reward (float) if the input is a string.
    """
    if isinstance(generated_sample, str):
        return float(generated_sample.count(char))
    else:
        return t.tensor([float(s.count(char)) for s in generated_sample])

def reward_to_judge(reward_fn: Callable[[str], float | int]) -> Callable[[Sequence[str], Sequence[str]], Int[Tensor, "batch"]]:
    """
    Converts a reward function to a judge function.
    """
    def judge_fn(generated_samples: Sequence[str]) -> Int[Tensor, "batch"]:
        rewards = reward_fn(generated_samples)
        return t.argmax(rewards, dim=0)
    return judge_fn
    
# %%
# On the fly dataloader

class OnTheFlyBinaryPreferenceDataset(t.utils.data.Dataset):
    def __init__(
            self, 
            prompt: str, 
            judge_fn: Callable[[Sequence[str], Sequence[str]], Int[Tensor, "batch"]],
            gen_model: DPOModel = ref_model, 
            tokenizer: PreTrainedTokenizer = tokenizer, 
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
        self.judge_fn = judge_fn
        self.gen_model = gen_model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.encoded_prompt = tokenizer(self.prompt, return_tensors="pt").to(device)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        preferred, rejected = self.generate_preference_pairs(batch_size=1)
        return {
            "preferred_column": self.tokenizer.decode(preferred[0], skip_special_tokens=True),
            "rejected_column": self.tokenizer.decode(rejected[0], skip_special_tokens=True)
        }
        
    @t.inference_mode()
    def generate_preference_pairs(
        self,
        batch_size: int,
        temperature: float = 1.0,
        max_length: int = args.gen_len,
        device: t.device = device
    ) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
        """Generate a batch of preferred and rejected completions using the reference model."""
        
        gen_tokens, gen_strings = self.gen_model.generate(
            self.encoded_prompt,
            max_length=max_length,
            batch_size=batch_size*2,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature
        )
        preferred_indices = self.judge_fn(gen_strings[:batch_size], gen_strings[batch_size:], tokenizer=tokenizer)
        # Reshape gen_tokens into (2, batch_size, seq_len)
        gen_tokens_reshaped = einops.rearrange(gen_tokens, "(b1 b2) ... -> b1 b2 ...", b1=2)
        
        # Gather preferred and rejected completions
        preferred_tokens = t.gather(gen_tokens_reshaped, 0, preferred_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, gen_tokens_reshaped.size(-1)))
        rejected_tokens = t.gather(gen_tokens_reshaped, 0, (1 - preferred_indices).unsqueeze(0).unsqueeze(-1).expand(-1, -1, gen_tokens_reshaped.size(-1)))
        
        # Squeeze to remove the extra dimension
        preferred_tokens = preferred_tokens.squeeze(0)
        rejected_tokens = rejected_tokens.squeeze(0)
        return preferred_tokens, rejected_tokens

# Create the on-the-fly dataset and dataloader
on_the_fly_dataset = OnTheFlyBinaryPreferenceDataset(prompt=args.prefix, num_samples=args.train_length)
on_the_fly_dataloader = t.utils.data.DataLoader(
    on_the_fly_dataset,
    batch_size=args.batch_size,
    collate_fn=lambda batch: collate_prompt_integrated(batch, tokenizer),
    num_workers=4
)


# %%
@dataclass
class DPOTrainer:
    model: DPOModel
    ref_model: DPOModel = field(default_factory=lambda: DPOModel())
    args: DPOTrainingArgs
    optimizer: t.optim.Optimizer | None = None
    scheduler: t.optim.lr_scheduler.LRScheduler | None = None

    def __post_init__(self):
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)

    def dpo_loss(
            self, 
            logits: Float[Tensor, "batch"], 
            ref_logits: Float[Tensor, "batch"],
            
            **kwargs
        ) -> Tensor:
        """
        Computes the DPO loss for a batch of logits and rewards.
        """
        raise NotImplementedError

# %%
if MAIN:
    print(dpo_model.generate("What is the capital of France?", batch_size=3))

# %%
