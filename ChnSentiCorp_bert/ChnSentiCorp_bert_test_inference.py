import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

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

# 3. 加載已保存的模型和分詞器
model = BertForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = BertTokenizer.from_pretrained("./sentiment_model")

# 將模型移動到指定設備
model.to(device)

# 4. 定義預測函數
def predict(text):
    """
    對單條文本進行情感分類。
    Args:
        text (str): 要進行分類的文本
    Returns:
        str: 預測的情感類別（Positive 或 Negative）
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
    label_map = {0: "Negative", 1: "Positive"}
    return label_map[predicted_class]

# 5. 對測試集逐條推理
test_texts = [item["text"] for item in dataset["test"]]
test_labels = [item["label"] for item in dataset["test"]]

predictions = []
for text in test_texts:
    prediction = predict(text)  # 使用定義的 predict 函數
    predictions.append(prediction)

# 6. 輸出測試結果
for i, (text, label, prediction) in enumerate(zip(test_texts, test_labels, predictions)):
    print(f"示例 {i + 1}:")
    print(f"文本: {text}")
    print(f"真實標籤: {'Positive' if label == 1 else 'Negative'}")
    print(f"模型預測: {prediction}")
    print("-" * 50)

print("測試集推理完成！")
