# %%
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from utils import device, DATA_DIR, SEED
# %%
sft_steps: int = 5000
model: str = "openai-community/gpt2-large"

# Load the full dataset
full_dataset = load_dataset("stanfordnlp/imdb", split="train")
# Shuffle the dataset and select the first sft_steps entries
shuffled_dataset = full_dataset.shuffle(seed=SEED)
dataset = shuffled_dataset.select(range(sft_steps))
# %%
try:
    wandb.init(project="sft-training", name="gpt2-large-sft")


    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=512,
        output_dir=DATA_DIR / "models" / "sft",
        num_of_sequences=sft_steps,
        logging_steps=10,
        report_to="wandb",
        run_name="sft-gpt2-large",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
finally:
    wandb.finish()  