# 使用 Hugging Face 的 Transformers 來實現 BERT 模型的訓練微調（fine-tuning），以進行垃圾郵件的辨識分類。
Using Hugging Face's Transformers to implement fine-tuning of the BERT model for classifying spam emails.

垃圾郵件的Dataset可以去Kaggle下載 >> 
[Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)。
## SMSSpamCollection_bert.py 訓練講解
### 資料準備：
從 "SMSSpamCollection" 檔案中讀取資料，並分為訓練集和驗證集。<br/>
將標籤轉換成模型可接受的格式，將 'ham' 改為 0，'spam' 改為 1。

    def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels:
                item["labels"] = torch.tensor(self.labels[idx])
            return item
                
因為val[idx]的idx只能接受數值，無法接收string，因此需要轉換型別。

### 模型準備：
使用 BERT 模型的預訓練版本 "bert-base-uncased"，透過 `BertForSequenceClassification` 來建立文本分類模型。<br/>
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

### 使用 tokenizer 轉換文字：
定義了一個自訂的 Dataset 類別，用來建立訓練集和驗證集的 Dataset。<br/>
使用 tokenizer 將文字轉換為模型可接受的輸入格式。

# 初始化 tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 將資料集分成訓練集和驗證集
X = list(df['message'])
y = list(df['label'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 使用 tokenizer 將文字轉換為模型可接受的輸入格式
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    # Create torch dataset # 定義自訂的 Dataset 類別
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
    
    # 建立訓練集和驗證集的 Dataset
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

**初始化 tokenizer 和模型**：使用 Hugging Face 的 Transformers 庫中的 `BertTokenizer` 和 `BertForSequenceClassification` 類別，從預訓練的 BERT 模型中初始化 tokenizer 和分類模型。`model_name` 指定了要使用的預訓練模型，這裡使用了 `bert-base-uncased`，這是一個英文模型

**準備資料集**：從讀取的資料中取出文本和標籤，然後使用 `train_test_split` 函式將資料集分成訓練集和驗證集，其中設置了驗證集佔總資料集的 20%。

**使用 tokenizer 轉換文字**：將訓練集和驗證集的文本資料使用 tokenizer 轉換成模型可接受的輸入格式。這包括將文本轉換成 token IDs，並進行 padding 和截斷，確保每個輸入序列的長度相同，這裡設置了最大長度為 512。

**建立自訂的 Dataset 類別**：定義了一個自訂的 `Dataset` 類別，用來封裝資料集，使其可以被 PyTorch 的 DataLoader 使用。該類別接受 tokenized 的資料和對應的標籤，並在 `__getitem__` 方法中將其轉換成 PyTorch 張量格式。

**建立訓練集和驗證集的 Dataset 物件**：將 tokenized 的訓練集和驗證集資料以及對應的標籤傳入自訂的 `Dataset` 類別，建立訓練集和驗證集的 Dataset 物件。

這樣做的目的是為了準備好訓練所需的資料格式，使其可以被 PyTorch 模型接受並用於訓練。

### 訓練模型：
定義了計算評估指標的函式，包括準確率（accuracy）、召回率（recall）、精確率（precision）、F1 分數（F1 score）。<br/>
初始化了 Trainer，設定了訓練相關的參數，包括訓練集、驗證集、計算評估指標的函式等。<br/>
使用 `trainer.train()` 開始訓練模型，同時設置了提前停止訓練的機制，以防止過度擬合。<br/>

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
{'loss': 0.0867, 'grad_norm': 0.03188520669937134, 'learning_rate': 3.506571087216249e-05, 'epoch': 0.9}<br/>
...<br/>
{'train_runtime': 3338.7943, 'train_samples_per_second': 4.005, 'train_steps_per_second': 0.501, 'train_loss': 0.03664293684953548, 'epoch': 3.0}<br/>
100%|----------------------------------------------------------------------------------| 1674/1674 [55:38<00:00, 1.99s/it]

![](https://github.com/weitsung50110/Bert_HugginFace_Train_Predict_SpamEmails/blob/main/github_imgs/train_img.png)

## SMSSpamCollection_bert_predict 預測講解
### 文本被tokenized後會變成什麼模樣 ：

    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    print(X_test_tokenized)

X_test_tokenized 裡面包含 Tokenization 的結果，每個樣本都包含了三個部分：`input_ids`、`token_type_ids` 和 `attention_mask`。

>{'input_ids': [[101, 2323, 1045, 3288, 1037, 5835, 1997, 4511, 2000, 2562, 2149, 21474, 1029, 2074, 12489, 999, 1045, 1521, 2222, 3288, 2028, 7539, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2131, 1996, 6745, 2718, 3614, 5524, 2005, 2489, 999, 11562, 2023, 4957, 1024, 8299, 1013, 1013, 1056, 5244, 1012, 2898, 3669, 3726, 1012, 4012, 1013, 5950, 1012, 1059, 19968, 1029, 8909, 1027, 26667, 24087, 2549, 4215, 2692, 27717, 19841, 24087, 2581, 22907, 14526, 1004, 2034, 1027, 2995, 1067, 1039, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

1.  `input_ids`: 這是每個文本樣本轉換後的 token 序列。每個 token 都對應到 BERT 模型的詞彙表中的一個索引。在這個例子中，每個樣本都包含了一個長度為 50 的 token 序列。如果某個樣本的 token 數量不足 50，則會使用 0 進行填充，直到達到指定的序列長度。

2.  `token_type_ids`: 這個部分是用來區分不同句子的。在這裡，所有的 token 都屬於同一個句子，因此對應的值都是 0。在處理文本對任務時，將會有兩個句子，並使用 0 和 1 來區分它們。

3.  `attention_mask`: 這個部分用來指示哪些 token 是模型在處理時應該關注的。在這裡，所有的 token 都是被處理的，因此對應的值都是 1。在填充的部分，對應的值則是 0，用於告訴模型這些部分是填充的，不應該參與計算。

這些 tokenization 結果是 BERT 模型在處理文本數據時所需的輸入格式，其中包括了文本的 token 序列、句子區分和注意力遮罩等信息。


### 預測結果：

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

`raw_pred` 為包含 9 個元素的一維數組。

[[ 4.478097  -4.762506 ] <br/> [-3.7398722  3.903498 ] <br/> [-3.7016623  3.8625014] <br/> [-3.7578042  3.9365413] <br/> [-3.6734304  3.8043854] <br/> [ 4.5997095 -4.8369007] <br/> [-3.3514545  3.4255216] <br/> [-3.7296603  3.8799422] <br/> [ 3.270107  -3.534067 ]]

`np.argmax` 函式返回一個包含 9 個元素的一維數組，其中每個元素是對應行的預測結果（0 或 1）。<br/>
如果第一個數值較大，則對應位置的元素為 0；如果第二個數值較大，則對應位置的元素為 1。

![](https://github.com/weitsung50110/Bert_HugginFace_Train_Predict_SpamEmails/blob/main/github_imgs/test_img.png)
