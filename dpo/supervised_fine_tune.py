from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from utils import device, DATA_DIR

dataset = load_dataset("stanfordnlp/imdb", split="train")

sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=512,
    output_dir=DATA_DIR / "models" / "sft",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()