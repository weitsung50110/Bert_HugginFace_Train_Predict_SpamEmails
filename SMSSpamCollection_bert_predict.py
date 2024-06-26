import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

# print(df.head())
# print(df.shape)
# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# ----- 3. Predict -----#
# Load test data
test_data  = pd.read_csv('SMSSpamCollection_test', sep='\t',
                           names=["message"])

# 從測試資料中提取特徵
X_test = list(test_data["message"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
print(X_test_tokenized)

# Create torch dataset # 創建 torch 資料集
test_dataset = Dataset(X_test_tokenized)

# Load trained model # 載入訓練好的模型
model_path = "output/checkpoint-1500"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer # 定義測試的 Trainer
test_trainer = Trainer(model)

# Make prediction # 進行預測
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions # 處理原始預測值
y_pred = np.argmax(raw_pred, axis=1)

print(y_pred)
