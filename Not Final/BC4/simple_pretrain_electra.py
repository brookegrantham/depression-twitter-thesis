import pandas as pd
from sklearn.model_selection import train_test_split
import os
from simpletransformers.classification import ClassificationModel
import logging
from simpletransformers.language_modeling import LanguageModelingModel


all_text = []

df = pd.read_table('reddit-pretrain-text', header=None)
df.columns = ["text"]
texts = df.text.tolist()
texts = [t for t in texts if isinstance(t, str)]
all_text.extend(texts)

train, test = train_test_split(all_text, test_size=0.1)

with open("train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 64,
    "eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "simple",
    "wandb_project": "Depressiom - ELECTRA",
    "wandb_kwargs": {"name": "Electra-SMALL"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 52000,
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
    },
}

train_file = "train.txt"
test_file = "test.txt"

model = LanguageModelingModel(
    "electra",
    None,
    args=train_args,
    train_files=train_file,
)

model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)


model = LanguageModelingModel(
    "electra",
    "outputs/best_model",
    args={"output_dir": "discriminator_trained"}
)

model.save_discriminator()


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_file = "reddit_text.txt"
labels = ['not', 'moderate', 'severe']

train_args = {
    "output_dir": "ner_output",
    "overwrite_output_dir": True,
}

model = ClassificationModel("electra", "discriminator_trained/discriminator_model", args=train_args, num_labels=3)

# Train the model
model.train_model(df)
result, model_outputs, predictions = model.eval_model(df)

print(result)