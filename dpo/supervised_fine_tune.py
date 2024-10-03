# %%
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from utils import device, DATA_DIR, SEED
# %%
sft_steps: int = 4000
model: str = "openai-community/gpt2-large"

# Load the full dataset
full_dataset = load_dataset("stanfordnlp/imdb", split="train")
# Shuffle the dataset and select the first sft_steps entries
shuffled_dataset = full_dataset.shuffle(seed=SEED)
train_val_split = shuffled_dataset.select(range(sft_steps)).train_test_split(test_size=0.05, seed=SEED)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
# %%
try:
    wandb.init(project="sft-training", name="gpt2-large-sft")


    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=512,
        output_dir=DATA_DIR / "models" / "sft",
        num_of_sequences=sft_steps,
        num_train_epochs=1.0,
        logging_steps=10,
        evaluation_strategy="steps",  # Evaluate at regular intervals
        eval_steps=50,  # Evaluate every 200 steps
        report_to="wandb",
        run_name="sft-gpt2-large-validation",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )
    trainer.train()
finally:
    wandb.finish()  