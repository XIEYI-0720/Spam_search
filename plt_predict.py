import numpy as np
import matplotlib.pyplot as plt
import random  # 导入random模块

# 指标名称
metrics = ['Accuracy', 'Precision', 'F1 Score']

# 模型名称
models = ['logistic','fasttext', 'bert', 'voting', 'stacking']

# 模型性能数据
data = {
    'Accuracy': [0.961,0.971, 0.994, 0.9931, 0.9949],
    'Precision': [0.961134,0.971019, 0.994003, 0.993130, 0.994908],
    'F1 Score': [0.960995,0.970999, 0.993999, 0.993099, 0.994899],
}

# 设置图形大小
fig, ax = plt.subplots(figsize=(10, 6))

# 柱形图的宽度
width = 0.2
#  , , ,, ,'#EE4431''#B4B4D5''#FC8002''#1663A9''#FAC7B3'
# 定义一组吸引人的颜色
colors = ['#8481BA','#CEDFEF', '#ADDB88','#FABB6E', '#92C2DD']

# 为每个模型随机选择不同的颜色，确保颜色不重复
# model_colors = random.sample(colors, len(models))

# 生成每个指标下模型性能的柱形图
for i, model in enumerate(models):
    performances = [data[metric][i] for metric in metrics]
    x = np.arange(len(metrics))  # 指标数量
    # 为每个模型分配一个颜色
    ax.bar(x + i*width, performances, width, label=model, color=colors[i])

# 添加标题和标签
ax.set_title('Model Performance on Different Metrics')
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_xticks(x + width * (len(models)/2 - 0.5))  # 调整x轴标签的位置，使其位于每组柱形图的中间
ax.set_xticklabels(metrics)
ax.legend(ncol=5)

# 在柱形图上添加数值标签
def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.6f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# 遍历每个条形图添加标签
for i in ax.containers:
    add_labels(ax, i)

plt.tight_layout(pad=1.0)  # 设置单倍行距  
plt.ylim(0.96, 1)  # 设置y轴的范围
# 显示图形
plt.show()