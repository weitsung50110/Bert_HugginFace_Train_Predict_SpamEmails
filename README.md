# 如何使用 Hugging Face 的 Transformers 庫來實現 BERT 模型的微調（fine-tuning），以進行文本分類任務。

### SMSSpamCollection_bert.py 訓練講解
### 資料準備：
從 "SMSSpamCollection" 檔案中讀取資料，並分為訓練集和驗證集。
將標籤轉換成模型可接受的格式，將 'ham' 改為 0，'spam' 改為 1。

    def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels:
                item["labels"] = torch.tensor(self.labels[idx])
            return item
                
因為val[idx]的idx只能接受數值，無法接收string，因此需要轉換型別。

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

根據Hugging Face官方文檔中，可以看到有非常多模型可以選擇，而本研究是使用google-bert/bert-base-uncased 

[Hugging Face BERT community](https://huggingface.co/google-bert)。

### 資料集準備：
定義了一個自訂的 Dataset 類別，用來建立訓練集和驗證集的 Dataset。
使用 tokenizer 將文字轉換為模型可接受的輸入格式。

### 訓練模型：
定義了計算評估指標的函式，包括準確率（accuracy）、召回率（recall）、精確率（precision）、F1 分數（F1 score）。
初始化了 Trainer，設定了訓練相關的參數，包括訓練集、驗證集、計算評估指標的函式等。
使用 trainer.train() 開始訓練模型，同時設置了提前停止訓練的機制，以防止過度擬合。

*   `output_dir`: 指定訓練過程中模型和日誌等輸出的目錄。
    
*   `evaluation_strategy`: 指定評估策略，這裡設置為 "steps"，表示基於步驟數進行評估。
    
*   `eval_steps`: 指定在訓練過程中每隔多少步進行一次評估。
    
*   `per_device_train_batch_size`: 每個訓練裝置（device）的批次大小。
    
*   `per_device_eval_batch_size`: 每個評估裝置的批次大小。
    
*   `num_train_epochs`: 訓練的總時代數（epochs）。
    
*   `seed`: 隨機種子，用於重現性。
    
*   `load_best_model_at_end`: 是否在訓練結束時載入最佳模型。
    
*   `logging_steps`: 每隔多少步輸出一次訓練日誌。
    
*   `report_to`: 指定將訓練進度報告到哪個工具，這裡設置為 "tensorboard"，表示將訓練進度報告到 TensorBoard。

        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            seed=0,
            load_best_model_at_end=True,
            logging_steps=10,  # 每 10 個步驟輸出一次訓練日誌
            report_to="tensorboard",
        )

`seed` 參數是用於設置隨機種子的，它的作用是確保在訓練過程中的隨機操作（例如參數初始化、數據順序洗牌等）是可重現的。通常情況下，當我們希望每次運行訓練過程時得到相同的結果時，就會設置隨機種子。這對於實驗的可重現性和結果的一致性非常重要。

`load_best_model_at_end` 參數用於控制是否在訓練結束時載入最佳的模型。在訓練過程中，模型的性能可能會隨著時間逐漸提升或者下降，因此通常會在每個評估步驟或者一定間隔之後進行模型性能的評估，並保存當前最佳的模型。當訓練結束時，這個參數可以確保載入最佳的模型，而不是最後一個模型，這樣可以確保我們得到的是在驗證集上性能最好的模型。

### 注意-如何把訓練結果print在cmd上
如果程式碼沒有加上以下兩個logging_steps, report_to="tensorboard"，你就會沒辦法馬上看到loss值，會變成一定要等Train完才能看到。

    args = TrainingArguments(
        logging_steps=10,  # 每 10 個步驟輸出一次訓練日誌
        report_to="tensorboard",
    )

>{'loss': 0.0528, 'grad_norm': 0.029315194115042686, 'learning_rate': 3.536439665471924e-05, 'epoch': 0.88}<br/>
{'loss': 0.0867, 'grad_norm': 0.03188520669937134, 'learning_rate': 3.506571087216249e-05, 'epoch': 0.9}
