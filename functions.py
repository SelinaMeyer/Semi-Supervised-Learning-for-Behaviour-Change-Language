import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from datasets import load_dataset, concatenate_datasets

TRAINING_ARGS = TrainingArguments("test_trainer", evaluation_strategy="epoch")
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

def show_confusion_matrix(conf_matrix, normalized=False, class_names=[0,1,2]):
    if normalized:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="PuRd")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    hmap= sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label');
    
def tokenize_function(examples):
    return tokenizer(examples["Sentence"], padding="max_length", truncation=True)

def compute_metrics(pred):
    logits, labels = pred
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'MCC': mcc
        #'support': support,
    }

def compute_test_metrics(pred, average):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': support,
        'MCC':mcc
    }

def compute_support(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    return {
        'support': support,
    }

def load_and_tokenize_training_set(filepath):
    vals_ds_bin = load_dataset('csv', data_files=filepath, split=[
        f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
        ])
    trains_ds_bin = load_dataset('csv', data_files=filepath, split=[
        f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
        ])
    
    for index, val_ds in enumerate(vals_ds_bin):
        val_ds = val_ds.remove_columns(["Unnamed: 0"])
        vals_ds_bin[index] = val_ds.map(tokenize_function, batched=True)
    
    for index, train_ds in enumerate(trains_ds_bin):
        train_ds = train_ds.remove_columns(["Unnamed: 0"])
        trains_ds_bin[index] = train_ds.map(tokenize_function, batched=True)
    
    return vals_ds_bin, trains_ds_bin

def load_predict_testset(data_files, model_path=None, trainer=None, args=TRAINING_ARGS):
    test = load_dataset('csv', data_files=data_files)
    test = test.remove_columns(["Unnamed: 0"])
    test = test.map(tokenize_function, batched=True)
    
    if model_path != None:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        trainer = Trainer(model=model, args=args)
        return trainer.predict(test["train"])
    else: 
        return trainer.predict(test["train"])
    
def load_tokenize_enhanced_set(data_files):
    test = load_dataset('csv', data_files=data_files, split="train")
    test = test.remove_columns(["Unnamed: 0"])
    test = test.map(tokenize_function, batched=True)
    return(test)

def predict_valence_with_confidence(data_files, pipe):
    ds = load_dataset('csv', data_files = data_files, split="train")
    #pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    label_0_scores = []
    label_1_scores = []
    for out in pipe(KeyDataset(ds, "split"), batch_size=8, truncation="only_first"):
        label_0_scores.append(out[0]['score'])
        label_1_scores.append(out[1]['score'])
    return(pd.DataFrame({"split": ds[:]['split'], "0":label_0_scores, "1": label_1_scores}))

def predict_labels_with_confidence(data_files, pipe):
    ds = load_dataset('csv', data_files = data_files, split="train")
    #pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    label_0_scores = []
    label_1_scores = []
    label_2_scores = []
    for out in pipe(KeyDataset(ds, "split"), batch_size=8, truncation="only_first"):
        label_0_scores.append(out[0]['score'])
        label_1_scores.append(out[1]['score'])
        label_2_scores.append(out[2]['score'])
    return(pd.DataFrame({"split": ds[:]['split'], "0":label_0_scores, "1": label_1_scores, "2": label_2_scores}))

def predict_sublabels_with_confidence(data_files, pipe):
    ds = load_dataset('csv', data_files = data_files, split="train")
    #pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    label_0_scores = []
    label_1_scores = []
    label_2_scores = []
    label_3_scores = []
    for out in pipe(KeyDataset(ds, "split"), batch_size=8, truncation="only_first"):
        label_0_scores.append(out[0]['score'])
        label_1_scores.append(out[1]['score'])
        label_2_scores.append(out[2]['score'])
        label_3_scores.append(out[3]['score'])
    return(pd.DataFrame({"split": ds[:]['split'], "0":label_0_scores, "1": label_1_scores, "2": label_2_scores, "3": label_3_scores}))


def finetune_more(dataset, conf_score, train_set, label_num, eval_set, training_args, mode=None):
    predict_1 = dataset[dataset["0"] > conf_score].copy()
    predict_1["labels"] = 0
    if mode == "confidence":
        if conf_score < 0.5:
            a_1 = dataset[dataset["1"] > (1-conf_score)].copy()
        else: 
            a_1 = dataset[dataset["1"] > conf_score].copy()
        a_1["labels"] = 1
        predict_1 = pd.concat([predict_1,a_1])
    if mode == "equal":
        a_1 = dataset[dataset["1"] > conf_score].copy()
        a_1["labels"] = 1
        a_1.sort_values("1", inplace=True, ascending=False)
        predict_1 = pd.concat([predict_1, a_1[:predict_1.shape[0]]])
    predict_1.rename(columns={"split":"Sentence"}, inplace=True)
    predict_1 = predict_1.sample(frac=1)
    predict_1.to_csv("enhanced_test_1.csv")
    enhanced_pred = load_tokenize_enhanced_set("enhanced_test_1.csv")
    enhanced = concatenate_datasets([train_set, enhanced_pred])
    enhanced = enhanced.shuffle()
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=label_num)
    trainer = Trainer(model=model, args=training_args, train_dataset=enhanced, eval_dataset=eval_set, compute_metrics=compute_metrics)
    trainer.train()
    return compute_test_metrics(trainer.predict(eval_set), None), compute_test_metrics(trainer.predict(enhanced), None)
    
def finetune_more_labels(dataset, conf_score, train_set, label_num, eval_set, training_args, mode=None):
    predict_1 = dataset[dataset["2"] > conf_score].copy()
    predict_1["labels"] = 2
    if mode == "confidence":
        if conf_score < 0.5:
            rest = dataset[dataset["2"] <= conf_score].copy()
            _1 = rest[rest["0"] > 0.5].copy()
            a_1 = rest[rest["1"] > 0.5].copy()
        else: 
            _1 = dataset[dataset["0"] > conf_score].copy()
            a_1 = dataset[dataset["1"] > conf_score].copy()
        _1["labels"] = 0
        a_1["labels"] = 1
        predict_1 = pd.concat([predict_1,a_1,_1])
    if mode == "equal":
        _1 = dataset[dataset["0"] > conf_score].copy()
        a_1 = dataset[dataset["1"] > conf_score].copy()
        _1["labels"] = 0
        a_1["labels"] = 1
        _1.sort_values("0", inplace=True, ascending=False)
        a_1.sort_values("1", inplace=True, ascending=False)
        predict_1 = pd.concat([predict_1, a_1[:predict_1.shape[0]], _1[:predict_1.shape[0]]])
    predict_1.rename(columns={"split":"Sentence"}, inplace=True)
    predict_1 = predict_1.sample(frac=1)
    predict_1.to_csv("enhanced_test_1.csv")
    enhanced_pred = load_tokenize_enhanced_set("enhanced_test_1.csv")
    enhanced = concatenate_datasets([train_set, enhanced_pred])
    enhanced = enhanced.shuffle()
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=label_num)
    trainer = Trainer(model=model, args=training_args, train_dataset=enhanced, eval_dataset=eval_set, compute_metrics=compute_metrics)
    trainer.train()
    return compute_test_metrics(trainer.predict(eval_set), None), compute_test_metrics(trainer.predict(enhanced), None)
    
def finetune_more_sublabels(dataset, conf_score, train_set, label_num, eval_set, training_args, mode=None):
    predict_1 = dataset[dataset["3"] > conf_score].copy()
    predict_1["labels"] = 3
    if mode == "confidence":
        if conf_score < 0.5:
            rest = dataset[dataset["3"] <= conf_score].copy()
            _1 = rest[rest["0"] > 0.5].copy()
            a_1 = rest[rest["1"] > 0.5].copy()
            d_1 = rest[rest["2"] > 0.5].copy()
        else: 
            _1 = dataset[dataset["0"] > conf_score].copy()
            a_1 = dataset[dataset["1"] > conf_score].copy()
            d_1 = dataset[dataset["2"] > conf_score].copy()
        _1["labels"] = 0
        a_1["labels"] = 1
        d_1["labels"] = 2
        predict_1 = pd.concat([predict_1,a_1,_1,d_1])
    if mode == "equal":
        _1 = dataset[dataset["0"] > conf_score].copy()
        a_1 = dataset[dataset["1"] > conf_score].copy()
        d_1 = dataset[dataset["2"] > conf_score].copy()
        _1["labels"] = 0
        a_1["labels"] = 1
        d_1["labels"] = 2
        _1.sort_values("0", inplace=True, ascending=False)
        a_1.sort_values("1", inplace=True, ascending=False)
        d_1.sort_values("2", inplace=True, ascending=False)
        predict_1 = pd.concat([predict_1, a_1[:predict_1.shape[0]], _1[:predict_1.shape[0]], d_1[:predict_1.shape[0]]])
    predict_1.rename(columns={"split":"Sentence"}, inplace=True)
    predict_1 = predict_1.sample(frac=1)
    predict_1.to_csv("enhanced_test_1.csv")
    enhanced_pred = load_tokenize_enhanced_set("enhanced_test_1.csv")
    enhanced = concatenate_datasets([train_set, enhanced_pred])
    enhanced = enhanced.shuffle()
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=label_num)
    trainer = Trainer(model=model, args=training_args, train_dataset=enhanced, eval_dataset=eval_set, compute_metrics=compute_metrics)
    trainer.train()
    return compute_test_metrics(trainer.predict(eval_set), None), compute_test_metrics(trainer.predict(enhanced), None)