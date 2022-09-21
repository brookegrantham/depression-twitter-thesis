import pandas as pd
import numpy as np
import torch
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer, ElectraForSequenceClassification,\
    AutoModelForSequenceClassification, RobertaForSequenceClassification

import gc

def empty_gpu():
    model = None
    gc.collect()
    torch.cuda.empty_cache()

def data_loader(filename, temp_size, test_size):

    df = pd.read_csv(filename)
    # df = df[:10]
    training_tweets, temp_tweets, training_labels, temp_labels = train_test_split(list(df['text']),
                                                                                  list(df['label']), test_size=temp_size,
                                                                                  random_state=123)
    val_tweets, test_tweets, val_labels, test_labels = train_test_split(temp_tweets, temp_labels, test_size=test_size,
                                                                        random_state=123)
    train = {'text': training_tweets, 'label': training_labels}
    val = {'text': val_tweets, 'label': val_labels}
    test = {'text': test_tweets, 'label': test_labels}

    train = Dataset.from_dict(train)
    val = Dataset.from_dict(val)
    test = Dataset.from_dict(test)

    return train, val, test


# tokenize function
def tokenize_function(dataset):
    model_inputs = tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=100)
    return model_inputs


def tokenize_set(train, val, test):
    train_dataset = train.map(tokenize_function, batched=True)
    val_dataset = val.map(tokenize_function, batched=True)
    test_dataset = test.map(tokenize_function, batched=True)
    return train_dataset, val_dataset, test_dataset

training_args = TrainingArguments(
    output_dir="transformer_checkpoints",
    per_device_train_batch_size=5,
    num_train_epochs=1,
    # auto_find_batch_size=True
)

def train_transformer(model,tok_train_dataset, tok_val_dataset):

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train_dataset,
        eval_dataset=tok_val_dataset,
    )

    trainer.train()
    return model

def predict_nn(trained_model, tok_test_dataset):

    output = trained_model(attention_mask=torch.tensor(tok_test_dataset["attention_mask"]), input_ids=torch.tensor(tok_test_dataset["input_ids"]))

    pred_labs = np.argmax(output["logits"].cpu().detach().numpy(), axis=1)

    gold_labs = tok_test_dataset["label"]

    return gold_labs, pred_labs


model_name = []
acc_scores = []
f1_scores = []
set = []

set_name=['reddit']

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
train, val, test = tokenize_set(tr, va, te)
model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=3)
for param in model.bert.parameters():
    param.requires_grad = False
trained_model = train_transformer(model, train, val)
gold, pred = predict_nn(trained_model, test)
model_name += ['prajjwal1/bert-tiny']
acc_scores += [accuracy_score(gold, pred)]
f1_scores += [f1_score(gold, pred, average='macro')]
set += set_name
empty_gpu()


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['bert-base-uncased']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['bert-base-cased']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
# # for param in model.roberta.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['roberta-base']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)
# # for param in model.roberta.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['roberta-large']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
# #
# tokenizer = BertTokenizer.from_pretrained('electra-base-discriminator')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-base-discriminator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-base-discriminator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
# #
# tokenizer = BertTokenizer.from_pretrained('electra-large-discriminator')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-large-discriminator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-large-discriminator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = BertTokenizer.from_pretrained('electra-base-generator')
# tr, va, te = data_loader('reddit_set.csv', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-base-generator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-base-generator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
# #
#
# set_name=['twitter']
#
# tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=3)
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['prajjwal1/bert-tiny']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['bert-base-uncased']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['bert-base-cased']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
# # for param in model.roberta.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['roberta-base']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)
# # for param in model.roberta.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['roberta-large']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
# #
# tokenizer = BertTokenizer.from_pretrained('electra-base-discriminator')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-base-discriminator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-base-discriminator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
# #
# tokenizer = BertTokenizer.from_pretrained('electra-large-discriminator')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-large-discriminator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-large-discriminator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()
#
# tokenizer = BertTokenizer.from_pretrained('electra-base-generator')
# tr, va, te = data_loader('all_tweets', 0.3, (1/3))
# train, val, test = tokenize_set(tr, va, te)
# model = ElectraForSequenceClassification.from_pretrained('electra-base-generator', num_labels=3)
# # for param in model.electra.parameters():
# #     param.requires_grad = False
# trained_model = train_transformer(model, train, val)
# gold, pred = predict_nn(trained_model, test)
# model_name += ['electra-base-generator']
# acc_scores += [accuracy_score(gold, pred)]
# f1_scores += [f1_score(gold, pred, average='macro')]
# set += set_name
# empty_gpu()

data = {'Model': model_name, 'Accuracy': acc_scores, 'Macro F1': f1_scores, 'Dataset:': set}
df = pd.DataFrame(data)
df.to_csv('baseline_results')
