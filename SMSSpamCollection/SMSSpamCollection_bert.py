import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

# 判斷是否有 CUDA，可用時使用 GPU，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---Using device: {device}")

# 讀取資料
df = pd.read_csv('SMSSpamCollection', sep='\t',
                 names=["label", "message"])

# 將 'ham' 改為 0，'spam' 改為 1
df['label'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)

# 定義使用的模型名稱
model_name = "bert-base-uncased"

# 初始化 tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 將資料集分成訓練集和驗證集
X = list(df['message'])
y = list(df['label'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 使用 tokenizer 將文字轉換為模型可接受的輸入格式
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512, return_tensors="pt")

# 定義自訂的 Dataset 類別
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# 建立訓練集和驗證集的 Dataset
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# 定義評估指標的計算函式
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# 定義 Trainer 的訓練參數
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=250,  # 過程中每隔多久用驗證集進行評估1次
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    logging_steps=10,  # 每 10 個步驟輸出一次訓練日誌
    report_to="tensorboard"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 開始訓練模型
trainer.train()
