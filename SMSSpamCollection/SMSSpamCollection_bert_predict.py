import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

# 判斷是否有 GPU，如果有則使用 GPU，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--Using device: {device}")

# 定義使用的模型名稱
model_name = "bert-base-uncased"

# 初始化 tokenizer（用於將文字轉換為模型所需格式）
tokenizer = BertTokenizer.from_pretrained(model_name)

# 定義自訂 Dataset 類別，用於處理測試資料
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        """
        初始化 Dataset
        encodings: 已 tokenized 的輸入數據
        labels: 如果有標籤，則傳入標籤；否則為 None
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        定義如何取得單筆數據
        idx: 數據索引
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        定義資料集的總數量
        """
        return len(self.encodings["input_ids"])

# ----- 預測階段 ----- #

# 讀取測試資料（只需輸入訊息部分）
test_data = pd.read_csv('SMSSpamCollection_test', sep='\t', names=["message"])

# 提取訊息內容並進行 tokenization
X_test = list(test_data["message"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(X_test_tokenized)

# 創建測試資料集
test_dataset = Dataset(X_test_tokenized)

# 載入已訓練好的模型，並將其移動到正確設備（GPU 或 CPU）
model_path = "output/checkpoint-1500"  # 訓練好的模型檔案位置
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

# 定義測試用的 Trainer（不需重新訓練模型）
test_trainer = Trainer(model=model)

# 將測試數據移動到正確設備
test_dataset.encodings = {key: val.to(device) for key, val in test_dataset.encodings.items()}

# 使用 Trainer 進行預測
raw_pred, _, _ = test_trainer.predict(test_dataset)
print(raw_pred)

# 處理原始預測結果，取得最終分類標籤（取每行預測的最大值索引）
y_pred = np.argmax(raw_pred, axis=1)

# 將預測結果與原始訊息結合，方便查看
test_data["prediction"] = y_pred  # 將預測結果加入測試資料 DataFrame 中

# 輸出預測結果
print(test_data)