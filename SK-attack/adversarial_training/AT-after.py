from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments,Trainer
import datasets
import pickle
def dataset_mapping(x):
    return{"x": x["text"][:512],
           "label":  1 if x["ended"] == True else 0,}
def tokenize_function(examples):
    return tokenizer(examples["x"],padding="max_length",truncation=True)

with open('adv5.pickle', 'rb') as f:
    data = pickle.load(f)
adversarial_samples = datasets.Dataset.from_dict(data)#对抗样本
origin_dataset = datasets.load_dataset('csv', data_files='small-117M-k40.test.csv', split='train[100:142]').map(function=dataset_mapping)
eval_dataset = datasets.load_dataset('csv', data_files='small-117M-k40.test.csv', split='train[150:178]').map(function=dataset_mapping)
new_dataset = {
        "x": [],
        "label": [],
    }  # 新数据集
for it in origin_dataset:
    new_dataset["x"].append(it["x"])
    new_dataset["label"].append(it["label"])

for it in adversarial_samples:
    new_dataset["x"].append(it["x"])
    new_dataset["label"].append(it["y"])

train_dataset = datasets.Dataset.from_dict(new_dataset)
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
#print("dataset:",dataset)
train_tokenized_datasets = train_dataset.map(tokenize_function,batched=True)
eval_tokenized_datasets = eval_dataset.map(tokenize_function,batched=True)
#print("tokenized_datasets:",tokenized_datasets)
my_train_dataset = train_tokenized_datasets
my_eval_dataset = eval_tokenized_datasets
model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")#检测器模型
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    #logits,labels=eval_pred
    #predictions = np.argmax(logits,axis=-1)
    #return metric.compute(predictions=predictions,references=labels)
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    acc = (labels == preds).sum()/len(labels)
    return {'accuracy':acc,}

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-3
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=my_train_dataset,
    eval_dataset=my_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
