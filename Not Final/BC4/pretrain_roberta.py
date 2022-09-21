from transformers import RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import os

# os.environ["WANDB_DISABLED"] = "true"

# tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# model = BertForMaskedLM.from_pretrained('prajjwal1/bert-tiny')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="reddit-pretrain-text",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./robertab-dep",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    seed=123
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

trainer.save_model("./robertab-dep")