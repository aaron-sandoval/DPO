# %%
import sys
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Union, Sequence, Any, Literal, Optional

import einops
import numpy as np
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
# from eindex import eindex
from jaxtyping import Float, Int, Bool
from torch import Tensor
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer, logging
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import device, DATA_DIR, SEED
from sentiment_judge import make_preference_pairs

logging.set_verbosity_error()
MAIN = __name__ == "__main__"



# %%


# %%

class DPOModel(nn.Module):
    def __init__(
            self, 
            model: str | Path,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            fp16: bool = False,
        ):
        super().__init__()
        if tokenizer is None:
            tokenizer = PreTrainedTokenizer.from_pretrained(model)
        if fp16:
            self.model = GPT2LMHeadModel.from_pretrained(model).half().to(device)
        else:
            
            self.model = GPT2LMHeadModel.from_pretrained(model).to(device)
        self.tokenizer = tokenizer
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id

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

# %%
# Reward and judge functions: simulating human preferences
def reward_char_count(sample: str, char: str = '.', *args, **kwargs) -> float:
    """Add 0.01 to keep away from wandb histogram bucket edges."""
    return sample.count(char) + 0.01

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
    def judge_fn(samples0: Sequence[str], samples1: Sequence[str]) -> Int[Tensor, "batch"]:
        rewards1 = t.tensor([reward_fn(s, *args, **kwargs) for s in samples1], requires_grad=False)
        rewards0 = t.tensor([reward_fn(s, *args, **kwargs) for s in samples0], requires_grad=False)
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
    base_model: str
    tokenizer: Optional[PreTrainedTokenizer] = None
    judge_fn: Callable[[Sequence[str], Sequence[str]], Bool[Tensor, "batch"]]
    implicit_reward_fn: Optional[Callable[[str], float | int]] = None

    # Basic / global
    seed: int = SEED
    cuda: bool = t.cuda.is_available()

    # Wandb / logging
    exp_name: str = "DPO"
    wandb_project_name: str | None = "capstone_dpo"
    wandb_entity: str | None = None  
    use_wandb: bool = True

    # Duration of different phases
    train_length: int = 64*400
    batch_size: int = 64

    # Optimization hyperparameters
    base_learning_rate: float = 1e-6  # Rafailov et al. 2024
    # head_learning_rate: float = 5e-4
    # max_grad_norm: float = 1.0
    warmup_steps: int = 150  # Rafailov et al. 2024
    final_scale: float = 0.1

    # Base model & sampling arguments
    gen_len: int = 30
    temperature: float = 0.6
    prefixes: list[str] = field(default_factory=lambda: ["This is"])

    # Extra stuff for DPO
    dpo_beta: float = 0.1
    judge_fn: Callable = judge_periods
    implicit_reward_fn: Callable = reward_char_count
    normalize_reward: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = PreTrainedTokenizer.from_pretrained(self.base_model)

# %%
def get_optimizer(args: DPOTrainingArgs, model: DPOModel) -> t.optim.Optimizer:
    """
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    """
    return t.optim.Adam(params=model.model.parameters(), lr = args.base_learning_rate)


# optimizer =' get_optimizer(args, dpo_model)

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
class OnTheFlySentimentPairDataset(t.utils.data.Dataset):
    def __init__(self, args: DPOTrainingArgs, gen_model: Optional[DPOModel], prefixes: Optional[list[str]], dataset: Optional[list[tuple[str]]] = None):
        self.args = args
        if dataset is None:
            self.num_samples = args.train_length
            self.gen_model = gen_model
            self.prefixes = prefixes
            self.dataset = []
            while len(self.dataset) < self.num_samples:
                self.generate_and_append_preference_pairs()
        else:
            self.dataset = dataset
            self.num_samples = len(self.dataset)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.dataser[idx]

    def generate_and_append_preference_pairs(
            self, 
            num_pairs_to_append: int = 24, 
            num_generations_per_prefix = 4) -> None:
        """
        Generates a batch of preference pairs and appends them to the dataset.
        """
        # Select subset of prefixes
        n_prefixes = num_pairs_to_append // (num_generations_per_prefix * (num_generations_per_prefix - 1) // 2)
        prefixes = random.sample(self.prefixes, n_prefixes)

        for prefix in prefixes:
            # Generate completions
            _, completions = self.gen_model.generate(
                prefix,
                batch_size=num_generations_per_prefix,
                gen_len=self.args.gen_len,
            )
            prefix_len = len(self.args.tokenizer.decode(prefix)["input_ids"])
            pairs = make_preference_pairs(completions)
            # Convert completions into preference pairs
            self.dataset.extend([{"preferred": p[0], "rejected": pairs[1], "prefix_len": prefix_len} for p in pairs])


        
    
# %%
# On the fly dataloader

class OnTheFlyBinaryPreferenceDataset(t.utils.data.Dataset):
    def __init__(
            self, 
            prompt: str, 
            args: DPOTrainingArgs,
            gen_model: DPOModel, 
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
        self.args = args
        self.prefix_len = len(args.tokenizer(prompt)["input_ids"])
        self.implicit_reward_fn = args.implicit_reward_fn
        self.judge_fn = args.judge_fn
        self.gen_model = gen_model
        self.num_samples = args.train_length
        self.encoded_prompt = args.tokenizer(self.prompt, return_tensors="pt").to(device)
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
        gen_len: Optional[int] = None,
        device: t.device = device
    ) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
        """Generate a batch of preferred and rejected completions using the reference model."""
        if gen_len is None:
            gen_len = self.args.gen_len
        gen_tokens, gen_strings = self.gen_model.generate(
            random.sample(self.args.prefixes, 1)[0],
            gen_len=gen_len,
            batch_size=batch_size*2,
            temperature=temperature,
        )
        preferred_index_0: Bool[Tensor, "batch"] = self.judge_fn(gen_strings[:batch_size], gen_strings[batch_size:])
        preferred_mask: Bool[Tensor, "2 batch"] = t.stack([preferred_index_0, ~preferred_index_0], dim=0).to(device)
        # Reshape gen_tokens into (2, batch_size, seq_len)
        gen_tokens_reshaped = einops.rearrange(gen_tokens, "(b1 b2) ... -> ... b1 b2", b1=2)
        # Gather preferred and rejected completions
        preferred_tokens = gen_tokens_reshaped.masked_select(preferred_mask)
        rejected_tokens = gen_tokens_reshaped.masked_select(~preferred_mask)
        preferred_tokens = einops.rearrange(preferred_tokens, "(seq batch) -> batch seq", batch=batch_size)
        rejected_tokens = einops.rearrange(rejected_tokens, "(seq batch) -> batch seq", batch=batch_size)
        return preferred_tokens, rejected_tokens



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
                pref_strs: list[str] = [self.args.tokenizer.decode(ids) for ids in batch["preferred"]]
                rej_strs: list[str] = [self.args.tokenizer.decode(ids) for ids in batch["rejected"]]
                gen_rewards = t.tensor([self.implicit_reward_fn(gen_str) for gen_str in pref_strs+rej_strs], dtype=t.float16, requires_grad=False)
                avg_pref_reward: float = sum(self.implicit_reward_fn(pref_str) for pref_str in pref_strs) / len(pref_strs)
                avg_rej_reward: float = sum(self.implicit_reward_fn(rej_str) for rej_str in rej_strs) / len(rej_strs)
                print(pref_strs[0])
                wandb.log({
                    "reward": {
                        "preferred": avg_pref_reward,
                        "rejected": avg_rej_reward,
                        "mean": gen_rewards.mean().item(),
                        "all": gen_rewards,
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
                    self.model.save_model()
                wandb.finish()
        else:
            self._train()
            if self.save_model:
                self.model.save_model()

# %%
if MAIN:
    args = DPOTrainingArgs(base_model="gpt2")
    dpo_model = DPOModel(model=args.base_model)
    ref_model = DPOModel(model=args.base_model, fp16=True)
    on_the_fly_dataloader = t.utils.data.DataLoader(OnTheFlyBinaryPreferenceDataset(
        args=args, 
        gen_model=dpo_model
    ))
    args.base_learning_rate = 4e-6
    args.warmup_steps = 50
    args.dpo_beta = 0.2
    args.use_wandb = True
    trainer = DPOTrainer(model=dpo_model, dataloader=on_the_fly_dataloader, ref_model=ref_model)
    trainer.train()

# %%
