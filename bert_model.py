import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import logging as transformers_logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns


print(datetime.datetime.now())
# 禁用transformers库的一些不必要的日志消息
transformers_logging.set_verbosity_error()
# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 10.5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus'] = False



# 加载数据
data = pd.read_csv("./data/data.csv")

# 准备数据
X = data["message"].tolist()
y = data["label"].tolist()

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=71)

# 定义BERT模型和tokenizer
model = BertForSequenceClassification.from_pretrained("./models/bert-base-chinese", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("./models/bert-base-chinese")

# 准备训练数据
train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", max_length=120)
val_encodings = tokenizer(X_val, padding=True, truncation=True, return_tensors="pt", max_length=120)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

del train_dataset, val_dataset, train_encodings, val_encodings ,X_train,X_val,y_train ,y_val# Delete variables that are no longer needed
torch.cuda.empty_cache()
# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 10)

# 启用混合精度训练
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备记录指标的CSV文件
with open('./results/bert.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['Epoch', 'Train Loss', 'val Loss', 'Train Accuracy', 'val Accuracy','train_precision','val_precision','train_recall','val_recall', 'Train F1', 'val F1'])


# 初始化最佳验证损失为正无穷大，用于早停和保存最佳模型
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 4  # 早停耐心值，连续4个epoch没有改善则停止训练

for epoch in range(10):  # 循环整个数据集

    model.train()
    train_loss, train_accuracy, train_f1 = 0, 0, 0
    all_train_preds, all_train_labels= [], []
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        train_loss += loss.item()
        
        # 移动到CPU进行计算指标
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=1)
        all_train_preds.extend(preds)
        all_train_labels.extend(label_ids)
        
    train_loss /= len(train_loader)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    train_precision = precision_score(all_train_labels, all_train_preds)
    train_recall=recall_score(all_train_labels, all_train_preds)
    train_f1 = f1_score(all_train_labels, all_train_preds)

    # print(f"Epoch {epoch+1}, Loss: {train_loss}, Accuracy: {train_accuracy}, F1: {train_f1}")
    
    # 在每个epoch结束时评估测试集
    model.eval()  # 设置模型为评估模式
    val_loss, val_accuracy, val_f1 = 0, 0, 0
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1)
            all_val_preds.extend(preds)
            all_val_labels.extend(label_ids)
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_recall=recall_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)
    # print(train_accuracy,train_recall,val_accuracy,val_recall)
    # 更新metrics.csv文件以包含指标
    with open('./results/bert.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_loss, train_accuracy, val_accuracy,train_precision,val_precision,train_recall,val_recall, train_f1, val_f1])
        file.close()
    #         # 在每个epoch后检查是否有改善
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0  # 重置早停计数器
        # 保存当前最佳模型
        model.save_pretrained(f"./models/111bert_best_model_epoch_{epoch+1}")
        print(f"Epoch {epoch+1}: Validation loss improved, saving model.")
        cm = confusion_matrix(all_val_labels, all_val_preds, labels=[0, 1])

                # 使用pandas将混淆矩阵转换为DataFrame，以便更好地显示标签
        cm_df = pd.DataFrame(cm, index=['实际: 0', '实际: 1'], columns=['预测: 0', '预测: 1'])

                # 使用seaborn绘制混淆矩阵热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.ylabel('实际值')
        plt.xlabel('预测值')
        plt.tight_layout(pad=1.0)  # 设置单倍行距  
        plt.savefig(f'./results/cm_epoch{epoch}.png')
        # plt.show()
    else:
        early_stopping_counter += 1
        print(f"Epoch {epoch+1}: No improvement in validation loss for {early_stopping_counter} consecutive epochs.")
    
    # 检查是否达到早停条件
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

# # 保存模型
model.save_pretrained("./models/bert_best_model")

# # 绘制训练集和测试集的学习曲线





# def index(preds,labels):    # 计算指标
#     acc = accuracy_score(labels, preds)
#     precision = precision_score(labels, preds, average='weighted')
#     f1 = f1_score(labels, preds, average='weighted')
#     print(f'准确率 = {acc}')
#     print(f'精确度 = {precision}')
#     print(f'F1 值 = {f1}')
#     # with open('./results/bert_predict_result.txt', 'a', encoding='utf-8') as f:
#     #     f.write(f'准确率 = {acc}\n')
#     #     f.write(f'精确度 = {precision}\n')
#     #     f.write(f'F1 值 = {f1}\n\n\n')
#     #     f.close()

# def bert_prob(texts, tokenizer, model):
#     model.eval()  # 设置为评估模式
#     predicted_probs = []
#     for text in texts:
#         inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             probs = torch.nn.functional.softmax(logits, dim=-1).numpy()  # 将logits转换为概率
#             format_probs = ["{:.16f}".format(num) for num in probs[0]]
#             format_probs=[float(n) for n in format_probs]
#             # print(format_probs)
#             predicted_probs.append(format_probs) 
#     return predicted_probs  # # ['0.0000081811', '0.9999917746']




# test_data = pd.read_csv('./data/test.csv')
# msg_test_list = [str(msg) for msg in test_data['message'].tolist()]
# label_test_list = test_data['label'].tolist()

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# bert_model = BertForSequenceClassification.from_pretrained("./models/bert_best_model_epoch_2")

# bert_test_preds = bert_prob(msg_test_list, tokenizer, bert_model)
# # print(bert_test_preds)
# final_train_voting_predictions = np.argmax(bert_test_preds, axis=1)
# # print(bert_test_preds[0])
# index(final_train_voting_predictions, label_test_list)
# print(datetime.datetime.now())