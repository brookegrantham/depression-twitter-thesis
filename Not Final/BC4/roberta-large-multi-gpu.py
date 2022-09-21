import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import get_linear_schedule_with_warmup, AdamW
from collections import defaultdict

imdb_df = pd.read_csv('reddit_set.csv', index_col = None)
imdb_df['review'] = imdb_df['text']
imdb_df = imdb_df.drop(['Unnamed: 0','PID','text'], axis =1)
PRETRAINED_MODEL_NAME = 'roberta-base'
roberta_model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=3)
roberta_tok = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(review,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              truncation=True,
                                              return_tensors='pt',
                                              return_token_type_ids=False,
                                              return_attention_mask=True,
                                              padding='max_length')

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


df_train, df_val = train_test_split(imdb_df, test_size = 0.2, random_state = 123)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CreateDataset(reviews=df.review.to_numpy(),
                       labels=df.label.to_numpy(),
                       tokenizer=tokenizer,
                       max_len=max_len
                       )

    return torch.utils.data.DataLoader(ds,
                                       batch_size=batch_size,
                                       num_workers=1,)


MAX_LEN = 248
BATCH_SIZE = 8

train_data_loader = create_data_loader(df_train, roberta_tok, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, roberta_tok, MAX_LEN, BATCH_SIZE)

check_data = next(iter(train_data_loader))
check_data.keys()


class MultiGPUClassifier(torch.nn.Module):
    def __init__(self, roberta_model):
        super(MultiGPUClassifier, self).__init__()
        self.embedding = roberta_model.roberta.embeddings.to('cuda:0')
        self.encoder = roberta_model.roberta.encoder.to('cuda:1')
        self.classifier = roberta_model.classifier.to('cuda:1')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        emb_out = self.embedding(input_ids.to('cuda:0'))
        enc_out = self.encoder(emb_out.to('cuda:1'))
        classifier_out = self.classifier(enc_out[0])
        return classifier_out


multi_gpu_roberta = MultiGPUClassifier(roberta_model)


EPOCHS = 4
LR = 2e-5

optimizer = AdamW(multi_gpu_roberta.parameters(), lr = LR)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps = 0,
                                           num_training_steps = total_steps)

loss_fn = torch.nn.CrossEntropyLoss().to('cuda:1')


def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    print(data_loader[0])
    for step,d in enumerate(data_loader):
        model.zero_grad()

        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}'.format(step, len(data_loader)))

        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1,
                                                         d['attention_mask'].shape[1])
        targets = d['labels']

        outputs = model(input_ids=input_ids, attention_mask=reshaped_attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets.to('cuda:1'))

        correct_predictions += torch.sum(preds == targets.to('cuda:1'))

        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1,
                                                             d['attention_mask'].shape[1])
            targets = d['labels']

            outputs = model(input_ids=input_ids, attention_mask=reshaped_attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets.to('cuda:1'))

            correct_predictions += torch.sum(preds == targets.to('cuda:1'))
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)



history = defaultdict(list)
best_accuracy = 0


for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_model(multi_gpu_roberta, train_data_loader, loss_fn, optimizer, scheduler,
                                        len(df_train))
    print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')

    val_acc, val_loss = eval_model(multi_gpu_roberta, val_data_loader, loss_fn, len(df_val))
    print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')

    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(multi_gpu_roberta.state_dict(), 'multi_gpu_roberta_best_model_state.bin')
        best_acc = val_acc


def get_predictions(model, data_loader):
    model = model.eval()
    texts_total = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            labels = d["labels"]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            texts_total.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
    return texts_total, predictions, prediction_probs, real_values


y_texts, y_pred, y_pred_probs, y_test = get_predictions(multi_gpu_roberta, val_data_loader)

class_names = ['Not depressed', 'Depressed', 'Severely']

f = open('roberta-large-CR', 'a')
f.write('\n')
f.write('roberta-base')
f.write(classification_report(y_test, y_pred, target_names=class_names))
f.write('\n')
f.close()

print(classification_report(y_test, y_pred, target_names=class_names))