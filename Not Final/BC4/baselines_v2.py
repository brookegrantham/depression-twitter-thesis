import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import gc

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")


def read_dataset(path):
    df = pd.read_csv(path, '\t')
    texts = df.text.values
    label_cats = df.label.astype('category').cat
    label_names = label_cats.categories
    labels = label_cats.codes

    print("Texts:", len(texts))
    print("Label names:", label_names)

    return texts, labels

def data_loader(filename, test_size):

    df = pd.read_csv(filename)
    # df = df[:10]
    training_tweets, val_tweets, training_labels, val_labels = train_test_split(list(df['text']),
                                                                                  list(df['label']), test_size=test_size,
                                                                                  random_state=123)
    train = {'text': training_tweets, 'label': training_labels}
    val = {'text': val_tweets, 'label': val_labels}

    return train, val


train,val = data_loader('reddit_set.csv', 0.2)


def run_model(model_name):
    max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    def tokenize_input(texts):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt')

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        return input_ids, attention_masks

    train_input_ids, train_attention_masks = tokenize_input(train['text'])
    val_input_ids, val_attention_masks = tokenize_input(val['text'])

    # convert everything to tensors

    train_input_ids = torch.cat(train_input_ids, dim=0)
    train_attention_masks = torch.cat(train_attention_masks, dim=0)
    train_labels = torch.tensor(train['label'], dtype=torch.long)

    val_input_ids = torch.cat(val_input_ids, dim=0)
    val_attention_masks = torch.cat(val_attention_masks, dim=0)
    val_labels = torch.tensor(val['label'], dtype=torch.long)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    batch_size = 8

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 3
    config.output_attentions = True

    print("config", config)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config)

    model.cuda()

    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    optimizer = AdamW(model.parameters(), lr=2e-5)  # args.learning_rate - default is 5e-5, our notebook had 2e-5)

    epochs = 4

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()

            if step % 40 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}'.format(step, len(train_dataloader)))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            total_train_loss += outputs.loss.item()

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("")
        print("Running Validation...")

        # torch.save(model, 'model_pickle_' + str(epoch_i))

        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        true_labels = []
        pred_labels = []
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                total_eval_loss += outputs.loss.item()
                logits = outputs.logits.detach().cpu().numpy().tolist()
                label_ids = b_labels.to('cpu').numpy().tolist()

                true_labels.extend(label_ids)
                pred_labels.extend(np.argmax(logits, axis=1))

        f = open('test', 'a')
        f.write('\n')
        f.write(model_name)
        f.write(str(epoch_i))
        f.write(str(accuracy_score(pred_labels, true_labels)))
        f.write(classification_report(pred_labels, true_labels))
        f.write('\n')
        f.close()


def empty_gpu():
    model = None
    gc.collect()
    torch.cuda.empty_cache()


run_model('prajjwal1/bert-tiny')
empty_gpu()
# run_model('bert-base-uncased')
# empty_gpu()
# run_model('bert-base-cased')
# empty_gpu()
# run_model('google/electra-base-discriminator')
# empty_gpu()
# run_model('google/electra-base-generator')
# empty_gpu()
# run_model('roberta-base')
# empty_gpu()

