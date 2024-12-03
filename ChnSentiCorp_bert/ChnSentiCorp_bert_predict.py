from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 選擇運算設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載已訓練的模型和分詞器
model_path = "./sentiment_model"  # 替換為模型保存的路徑
tokenizer = BertTokenizer.from_pretrained(model_path)  # 加載保存的分詞器
model = BertForSequenceClassification.from_pretrained(model_path)  # 加載保存的分類模型
model.to(device)  # 將模型移動到 GPU 或 CPU 設備

# 定義預測函數
def predict(text):
    """
    將輸入文本進行分詞處理，並通過模型進行分類預測。

    Args:
        text (str): 要預測的文本

    Returns:
        str: 預測的情感類別（"Positive" 或 "Negative"）
    """
    # 將文本轉換為模型所需的輸入格式
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # 將分詞後的數據移動到設備（GPU 或 CPU）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 通過模型進行推理，獲取分類結果
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()  # 取得分數最高的類別索引

    # 將類別索引映射為具體的情感標籤
    label_map = {0: "Negative", 1: "Positive"}
    return label_map[predicted_class]

# 命令列循環輸入文本進行預測
print("請輸入文本進行情感分析（輸入 'exit' 結束程式）：")
while True:
    # 接收使用者輸入
    test_text = input("輸入文本：")

    # 如果使用者輸入 'exit'，退出迴圈
    if test_text.lower() == "bye":
        print("程式結束，再見！")
        break

    # 將使用者輸入文本進行預測並打印結果
    print(f"Text: {test_text}\nPrediction: {predict(test_text)}")
