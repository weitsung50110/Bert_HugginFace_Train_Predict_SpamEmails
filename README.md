## 如何使用 Hugging Face 的 Transformers 庫來實現 BERT 模型的微調（fine-tuning），以進行文本分類任務。


### 資料準備：
從 "SMSSpamCollection" 檔案中讀取資料，並分為訓練集和驗證集。
將標籤轉換成模型可接受的格式，將 'ham' 改為 0，'spam' 改為 1。

### 模型準備：
使用 BERT 模型的預訓練版本 "bert-base-uncased"，透過 BertForSequenceClassification 來建立文本分類模型。
初始化 tokenizer，將文本轉換成模型可接受的輸入格式。

| Model                                       | #params | Language  |
|---------------------------------------------|---------|-----------|
| bert-base-uncased                           | 110M    | English   |
| bert-large-uncased                          | 340M    | English   |
| bert-base-cased                             | 110M    | English   |
| bert-large-cased                            | 340M    | English   |
| bert-base-chinese                           | 110M    | Chinese   |
| bert-base-multilingual-cased                | 110M    | Multiple  |
| bert-large-uncased-whole-word-masking       | 340M    | English   |
| bert-large-cased-whole-word-masking         | 340M    | English   |

根據huggingface官方文檔中，可以看到有非常多模型可以選擇，而本研究是使用google-bert/bert-base-uncased 。


### 資料集準備：
定義了一個自訂的 Dataset 類別，用來建立訓練集和驗證集的 Dataset。
使用 tokenizer 將文字轉換為模型可接受的輸入格式。

### 訓練模型：
定義了計算評估指標的函式，包括準確率（accuracy）、召回率（recall）、精確率（precision）、F1 分數（F1 score）。
初始化了 Trainer，設定了訓練相關的參數，包括訓練集、驗證集、計算評估指標的函式等。
使用 trainer.train() 開始訓練模型，同時設置了提前停止訓練的機制，以防止過度擬合。

### 注意
如果程式碼沒有加上以下兩個logging_steps, report_to="tensorboard"，你就會沒辦法馬上看到loss值，會變成一定要等Train完才能看到。

    args = TrainingArguments(
        logging_steps=10,  # 每 10 個步驟輸出一次訓練日誌
        report_to="tensorboard",
    )

>{'loss': 0.0528, 'grad_norm': 0.029315194115042686, 'learning_rate': 3.536439665471924e-05, 'epoch': 0.88}<br/>
{'loss': 0.0867, 'grad_norm': 0.03188520669937134, 'learning_rate': 3.506571087216249e-05, 'epoch': 0.9}
