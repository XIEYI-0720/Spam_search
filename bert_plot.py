from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('./results/bert_metrics.csv')  # 请替换为实际的文件路径

# 绘制训练和验证损失
ax = plt.figure(figsize=(10, 4)).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
plt.plot(data['Epoch'], data['val Loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,0.15)
plt.legend()
plt.tight_layout()
plt.show()
# 绘制训练和验证准确率

ax = plt.figure(figsize=(10, 4)).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy')
plt.plot(data['Epoch'], data['val Accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.97,1)
plt.legend()
plt.tight_layout()
plt.show()

# 绘制训练和验证F1分数（如果需要）
ax = plt.figure(figsize=(10, 4)).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(data['Epoch'], data['Train F1'], label='Train F1')
plt.plot(data['Epoch'], data['val F1'], label='Validation F1')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim(0.97,1)
plt.legend()
plt.show()