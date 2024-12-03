import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 選擇運算設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的設備: {device}")

# 2. 加載本地數據集
data_files = {
    "train": "./ChnSentiCorp/data/train-00000-of-00001-02f200ca5f2a7868.parquet",
    "validation": "./ChnSentiCorp/data/validation-00000-of-00001-405befbaa3bcf1a2.parquet",
    "test": "./ChnSentiCorp/data/test-00000-of-00001-5372924f059fe767.parquet",
}

dataset = load_dataset("parquet", data_files=data_files)

# 查看數據集結構
print(dataset)

# 3. 加載 BERT 分詞器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# 將模型移動到指定的設備
model.to(device)

# 4. 數據預處理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 對數據集進行分詞處理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. 設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",             # 訓練結果保存目錄
    evaluation_strategy="epoch",        # 每個 epoch 驗證一次
    learning_rate=2e-5,                 # 學習率
    per_device_train_batch_size=16,     # 每個設備的訓練批次大小
    per_device_eval_batch_size=16,      # 每個設備的驗證批次大小
    num_train_epochs=3,                 # 訓練輪次
    weight_decay=0.01,                  # 權重衰減
    logging_dir="./logs",               # 日誌目錄
    save_strategy="epoch",              # 每個 epoch 保存模型
    logging_steps=10                    # 每 10 步記錄一次
)

# 6. 定義評估函數
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 開始訓練
trainer.train()

# 保存模型
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# 8. 測試模型
def predict(text):
    """
    對單條文本進行情感分類。
    Args:
        text (str): 要進行分類的文本
    Returns:
        str: 預測的情感類別（Positive 或 Negative）
    """
    # 將文本轉換為模型輸入格式
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    # 將輸入移動到指定設備（GPU 或 CPU）
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 模型推理，獲取分類結果
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()  # 獲取分數最高的類別索引
    
    # 定義情感標籤映射
    label_map = {0: "Negative", 1: "Positive"}
    return label_map[predicted_class]

# 測試示例
test_text = "這個產品非常好，我非常喜歡！"
print(f"文本: {test_text}\n預測: {predict(test_text)}")
