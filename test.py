import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import logging as transformers_logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 禁用transformers库的一些不必要的日志消息
transformers_logging.set_verbosity_error()

# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_csv("./data/example.csv")

# 准备数据
X = data["message"].tolist()
y = data["label"].tolist()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)

# 定义BERT模型和tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 准备训练数据
train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", max_length=128)
test_encodings = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt", max_length=128)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

# 启用混合精度训练
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备记录指标的CSV文件
with open('./metrics.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Training F1'])


train_loss_set = []
train_accuracy_set = []
train_f1_set = []
test_loss_set = []
test_accuracy_set = []
test_f1_set = []
for epoch in range(3):  # 循环整个数据集3次
    model.train()
    total_loss = 0
    train_preds = []
    train_labels = []
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
        
        total_loss += loss.item()
        
        # 移动到CPU进行计算指标
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=1)
        train_preds.extend(preds)
        train_labels.extend(label_ids)
        
    train_loss = total_loss / len(train_loader)
    train_loss_set.append(train_loss)
    
    # 计算准确率和F1分数
    train_accuracy = accuracy_score(train_labels, train_preds)
    _, _, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='weighted')
    train_accuracy_set.append(train_accuracy)
    train_f1_set.append(train_f1)
    
    
    print(f"Epoch {epoch+1}, Loss: {train_loss}, Accuracy: {train_accuracy}, F1: {train_f1}")
    
    # 将指标写入CSV文件
    with open('./metrics.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, train_accuracy, train_f1])        



    # 在每个epoch结束时评估测试集
    model.eval()  # 设置模型为评估模式
    test_loss, test_accuracy, test_f1 = 0, 0, 0
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            test_loss += loss.item()
            
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1)
            all_test_preds.extend(preds)
            all_test_labels.extend(label_ids)
    
    test_loss /= len(test_loader)
    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    _, _, test_f1, _ = precision_recall_fscore_support(all_test_labels, all_test_preds, average='weighted')
    
    test_loss_set.append(test_loss)
    test_accuracy_set.append(test_accuracy)
    test_f1_set.append(test_f1)
    
    # 更新metrics.csv文件以包含测试集的指标
    with open('./metrics.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, test_loss, test_accuracy, test_f1])

# 保存模型
model.save_pretrained("./best_model")

# 绘制训练集和测试集的学习曲线
plt.figure(figsize=(10,5))
plt.title("Training and Testing Metrics")
plt.plot(train_loss_set, label='Train Loss')
plt.plot(test_loss_set, label='Test Loss')
plt.plot(train_accuracy_set, label='Train Accuracy')
plt.plot(test_accuracy_set, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.legend()
plt.show()



# # 加载模型
# model = BertForSequenceClassification.from_pretrained("./best_model")

# # 将模型设置为评估模式
# model.eval()

# # 假设你有一个待预测的文本
# text_to_predict = "这是一个示例文本。"
# encoded_input = tokenizer(text_to_predict, padding=True, truncation=True, max_length=128, return_tensors="pt")

# # 如果你有GPU，把输入和模型都放到GPU上
# if torch.cuda.is_available():
#     encoded_input = encoded_input.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     outputs = model(**encoded_input)
#     predictions = torch.argmax(outputs.logits, dim=1)

# # predictions是预测结果的tensor
# print(predictions)