import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 10.5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus']=False 
## 定义颜色 '#EE4431''#B4B4D5''#FC8002''#1663A9''#FAC7B3''#8481BA','#CEDFEF', '#ADDB88','#FABB6E', '#92C2DD'


# # # # -------------------------------------------------------两类标签下短信的长度
# # # # 读取CSV文件
df = pd.read_csv("./data/use_data.csv")
msg_column = df["message"]
labels = df["label"]  # 假设标签列名为"label"
sms_lengths_label=msg_column.str.len()
sms_lengths_label_0 = msg_column[labels == 0].str.len()
sms_lengths_label_1 = msg_column[labels == 1].str.len()

# # 使用seaborn绘制两类标签下短----------------------------------信长度的密度图
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.kdeplot(sms_lengths_label_0, shade=True, color="b", label="Label 0", bw_adjust=0.5)
sns.kdeplot(sms_lengths_label_1, shade=True, color="r", label="Label 1", bw_adjust=0.5)
plt.xlabel("message length")
plt.ylabel("density")
plt.legend()
plt.tight_layout(pad=1.0)  # 设置单倍行距  
plt.xlim(0, 120)
plt.show()

# # # -------------------------------------------------创建长度分组


# bins = np.arange(0, sms_lengths_label.max() + 100, 100)  # 根据最大长度确定分组的边界
# sms_lengths_label_groups = pd.cut(sms_lengths_label, bins)
# bins0 = np.arange(0, sms_lengths_label_0.max() + 100, 100)  # 根据最大长度确定分组的边界
# sms_lengths_label_0_groups = pd.cut(sms_lengths_label_0, bins0)
# bins1 = np.arange(0, sms_lengths_label_1.max() + 100, 100)  # 根据最大长度确定分组的边界
# sms_lengths_label_1_groups = pd.cut(sms_lengths_label_1, bins1)

# # 计算每个分组的短信数量
# sms_lengths_counts = sms_lengths_label_groups.value_counts().sort_index()
# sms_lengths_counts0 = sms_lengths_label_0_groups.value_counts().sort_index()
# sms_lengths_counts1 = sms_lengths_label_1_groups.value_counts().sort_index()

# # 打印每个分组的短信数量
# print(sms_lengths_counts)
# print(sms_lengths_counts0)
# print(sms_lengths_counts1)
# sms_lengths_counts.to_csv('length_count.csv', header=False)


# # # # # -------------------------------------------------------预处理后短信的长度变化
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('./data/data.csv')
# df =df.head(100)
# # Calculate the length of each message
# df['message_length'] = df['message'].apply(len)
# df['msg_new_length'] = df['msg_new'].apply(len)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['message_length'], label='Original Message Length')
# plt.plot(df.index, df['msg_new_length'], label='New Message Length', linestyle='--')
# plt.xlabel('Message Index')
# plt.ylabel('Length of Message')
# plt.legend()
# plt.tight_layout(pad=1.0)  # 设置单倍行距  

# plt.grid(True)
# plt.tight_layout()

# # Show plot
# plt.show()


# # # # # -------------------------------------------------------短信的影响因素特征矩阵
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re

# # # 假设df是您的DataFrame
# df = pd.read_csv('./data/use_data.csv')  # 加载数据

# # 定义函数计算特征
# def count_digits(text):
#     return sum(c.isdigit() for c in text)

# def count_symbols(text):
#     return sum(not c.isalnum() for c in text)

# def count_non_chinese(text):
#     return sum(not '\u4e00' <= c <= '\u9fff' for c in text)

# # 应用函数计算特征
# df['digit_count'] = df['message'].apply(count_digits)
# df['symbol_count'] = df['message'].apply(count_symbols)
# df['length'] = df['message'].apply(len)
# df['non_chinese_ratio'] = df['message'].apply(count_non_chinese) / df['length']

# # 计算相关系数矩阵
# corr_matrix = df[['digit_count', 'symbol_count', 'length', 'non_chinese_ratio']].corr()
# english_labels = ['数字长度', '符号个数', '短信长度', '非中文字符占比']

# # 绘制热图
# plt.figure(figsize=(7, 6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.xticks(np.arange(len(english_labels)) + 0.5, english_labels)
# plt.yticks(np.arange(len(english_labels)) + 0.5, english_labels)
# plt.tight_layout(pad=1.0)  # 设置单倍行距  
# plt.show()


# # ——--------------------------------------------------------------词云图
# from wordcloud import WordCloud
# import pandas as pd
# import matplotlib.pyplot as plt

# # 加载数据
# data = pd.read_csv('./data/use_data.csv')

# # 假设数据集中有一个名为'label'的列和一个名为'msg_new'的文本列
# unique_labels = data['label'].unique()

# for label in unique_labels:
#     # 为每个标签生成一个词云
#     subset = data[data['label'] == label]  # 根据标签筛选数据
#     texts = subset['msg_new']
#     texts_str = ' '.join(texts.astype(str))  # 将文本合并为一个长字符串
    
#     # 生成词云
#     wc = WordCloud(font_path="C:/Windows/Fonts/simsun.ttc", width=800, height=600, mode="RGBA", background_color=None).generate(texts_str)
    
#     # 显示词云
#     plt.figure(figsize=(10, 8))  # 设置图像大小
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")  # 不显示坐标轴
#     plt.tight_layout(pad=1.0)  # 设置单倍行距  
#     # wc.to_file(f"./data_wordcloud{label}.png")    
#     plt.show()
# # 保存词云图

# # # ——------------------------------------------------------准确率和AUC值--------指标直方图
# import matplotlib.pyplot as plt
# import numpy as np

# models = ['Logistic', 'FastText', 'BERT', 'Voting', 'Stacking']
# accuracy = [0.9646, 0.9709333333333333, 0.9951333333333333, 0.99, 0.9954]
# auc_values = [0.99162685, 0.9849616400000001, 0.99944438, 0.99891606, 0.9991305400000001]

# x = np.arange(len(models))
# width = 0.35

# fig, ax1 = plt.subplots()

# ax1.bar(x - width/2, accuracy, width, label='Accuracy', color='#FC8002')
# ax1.bar(x + width/2, auc_values, width, label='AUC Value', color='#1663A9')

# ax1.set_xlabel('Models')
# ax1.set_ylabel('Scores')
# ax1.set_xticks(x)
# ax1.set_xticklabels(models)
# ax1.legend()
# plt.ylim(0.9,1)
# plt.tight_layout(pad=1.0)  # 设置单倍行距  
# plt.tight_layout()
# plt.show()



# # # # ——------------------------------------------------------精确率、召回率、F1分数--------指标直方图
# import matplotlib.pyplot as plt
# import numpy as np

# models = ['Logistic', 'FastText', 'BERT', 'Voting', 'Stacking']
# x = np.arange(len(models))
# width = 0.25
# precision = [0.9446766169154229, 0.9534976152623211, 0.995375025135733, 0.9904935275080906, 0.9957771968630605]
# recall = [0.9494, 0.9596, 0.99, 0.9794, 0.9904]
# f1_scores = [0.9470324189526185, 0.9565390749601276, 0.9926802366389251, 0.9849155269509252, 0.9930813195628196]

# fig, ax2 = plt.subplots()

# ax2.bar(x - width, precision, width, label='Precision', color='#FC8002')
# ax2.bar(x, recall, width, label='Recall', color='#1663A9')
# ax2.bar(x + width, f1_scores, width, label='F1 Score', color='#8481BA')

# ax2.set_xlabel('Models')
# ax2.set_ylabel('Scores')
# ax2.set_xticks(x)
# ax2.set_xticklabels(models)
# ax2.legend()
# plt.ylim(0.9,1)
# plt.tight_layout(pad=1.0)  # 设置单倍行距  
# plt.tight_layout()
# plt.show()
# # # ——------------------------------------------------------耗时、准确率--------指标直方图
# import matplotlib.pyplot as plt
# import numpy as np

# # 模型和数据
# models = ['Logistic', 'FastText', 'BERT', 'Voting', 'Stacking']
# x = np.arange(len(models))
# width = 0.25
# training_time = [1383, 285, 10528, 0, 1065]
# prediction_time = [1, 1, 798, 799, 800]
# accuracy = [0.9646, 0.970933, 0.995133, 0.99, 0.9954]

# # 创建图和主要坐标轴
# fig, ax1 = plt.subplots()

# # 绘制训练耗时和预测耗时柱状图
# ax1.bar(x - width/2, training_time, width, label='训练耗时', color='#FC8002')
# ax1.bar(x + width/2, prediction_time, width, label='预测耗时', color='#1663A9')
# ax1.set_xlabel('Models')
# ax1.set_ylabel('Time (s)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(models)
# ax1.legend(loc='upper left')

# # 创建共享x轴的次坐标轴
# ax2 = ax1.twinx()
# ax2.plot(models, accuracy, label='准确率', color='#8481BA', marker='o', linestyle='-', linewidth=2)
# ax2.set_ylabel('Accuracy')
# ax2.legend(loc='upper right')
# plt.ylim(0.96,1)
# plt.tight_layout(pad=1.0)  # 设置单倍行距
# plt.tight_layout()
# plt.show()  



# # # ——------------------------------------------------------混淆矩阵
# # 假设y_true是你的真实标签，y_pred是模型的预测结果
# data = pd.read_csv('./data/test.csv')
# y_true = data['label'].to_list()
# y_pred = pd.read_csv('./results/predict_resultstacking.csv', header=None).squeeze().tolist()

# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)

# # 使用seaborn绘制混淆矩阵热图
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()




#-----------------------------------------------------------------BERT训练验证准确率和损失
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the CSV data
df = pd.read_csv('./bert/bert.csv')

# Plotting Accuracy
plt.figure(figsize=(7, 6))
plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
plt.plot(df['Epoch'], df['val Accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout(pad=1.0)  # 设置单倍行距
plt.ylim(0.92,1)
plt.show()

# Plotting Loss
plt.figure(figsize=(7, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['val Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout(pad=1.0)  # 设置单倍行距
plt.ylim(0,0.28)
plt.show()


# def plot_mix(model,val_preds,val_labels):
#     cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])

#                 # 使用pandas将混淆矩阵转换为DataFrame，以便更好地显示标签
#     cm_df = pd.DataFrame(cm, index=['实际: 0', '实际: 1'], columns=['预测: 0', '预测: 1'])

#                 # 使用seaborn绘制混淆矩阵热图
#     plt.figure(figsize=(5, 3))
#     sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
#     plt.ylabel('实际值')
#     plt.xlabel('预测值')
#     plt.tight_layout(pad=1.0)  # 设置单倍行距  
#     # plt.savefig(f'./sv/cm_model_{model}.png')
#     plt.show()
#     return 0


# val_preds = pd.read_csv('./sv/predictlogistic.csv', header=None).squeeze().tolist()
# data = pd.read_csv('./data/test.csv')
# val_labels = data['label'].tolist()
# plot_mix('bert',val_preds,val_labels)

