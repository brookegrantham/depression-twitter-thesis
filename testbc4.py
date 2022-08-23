import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer,TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import Dataset

df = pd.read_csv('reduced_set')
df['label'] = df['label'].replace({2:1})
all_tweets_df = df[:100]

# Training = 0.7, validation = 0.1, test = 0.2
training_tweets, temp_tweets, training_labels, temp_labels = train_test_split(list(all_tweets_df['text']), list(all_tweets_df['label']), test_size=0.3, random_state=123)
val_tweets, test_tweets, val_labels, test_labels = train_test_split(temp_tweets, temp_labels, test_size=(1/3), random_state=123)


tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')


def tokenize_function(dataset):
    model_inputs = tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=100)
    return model_inputs


train = {'text': training_tweets, 'label': training_labels}
val = {'text': val_tweets, 'label': val_labels}
test = {'text': test_tweets, 'label': test_labels}

train_dataset = Dataset.from_dict(train)
val_dataset = Dataset.from_dict(val)
test_dataset = Dataset.from_dict(test)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="transformer_checkpoints",
    num_train_epochs=1,
)

model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)

for param in model.bert.parameters():
    param.requires_grad = False


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


def predict_nn(trained_model, test_dataset):

    output = trained_model(attention_mask=torch.tensor(test_dataset["attention_mask"]), input_ids=torch.tensor(test_dataset["input_ids"]))

    pred_labs = np.argmax(output["logits"].detach().numpy(), axis=1)

    gold_labs = test_dataset["label"]

    return gold_labs, pred_labs


gold_labs, pred_labs = predict_nn(model, test_dataset)

df = pd.DataFrame([accuracy_score(pred_labs, gold_labs),f1_score(pred_labs,gold_labs)])
df.to_csv('testcsv')
# print("BERT-tiny on the John Hopkins Twitter dataset:")
# print(f'The accuracy score is {accuracy_score(pred_labs, gold_labs)}')
# print(f'The f1-score is {f1_score(pred_labs,gold_labs)}')
