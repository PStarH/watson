import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# 1. 读取数据
train_df = pd.read_csv('./data/contradictory-my-dear-watson/train.csv')
test_df = pd.read_csv('./data/contradictory-my-dear-watson/test.csv')
sample_submission_df = pd.read_csv('/mnt/data/sample_submission.csv')

# 2. 预处理数据
train_texts = list(zip(train_df['premise'], train_df['hypothesis']))
train_labels = train_df['label'].tolist()

test_texts = list(zip(test_df['premise'], test_df['hypothesis']))

# 3. 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 4. 数据编码
def encode_texts(texts, tokenizer, max_length=128):
    premises, hypotheses = zip(*texts)
    encodings = tokenizer(list(premises), list(hypotheses), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings

train_encodings = encode_texts(train_texts, tokenizer)
test_encodings = encode_texts(test_texts, tokenizer)

# 5. 创建PyTorch数据集
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = NLIDataset(train_encodings, train_labels)
test_dataset = NLIDataset(test_encodings)

# 6. 训练模型
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 7. 预测
predictions = trainer.predict(test_dataset).predictions.argmax(-1)

# 8. 保存预测结果
sample_submission_df['label'] = predictions
sample_submission_df.to_csv('/mnt/data/submission.csv', index=False)
