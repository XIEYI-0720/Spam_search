import csv
import joblib
import numpy as np
import pandas as pd
import fasttext
from sklearn.linear_model import LogisticRegression
import datetime
from sklearn.model_selection import GridSearchCV, learning_curve
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, BertTokenizer
import seaborn as sns



# Improved settings for global font and size
plt.rcParams.update({'font.sans-serif': 'SimSun', 'font.size': 10.5, 'axes.unicode_minus': False})

def compute_metrics(model,preds, labels):
    """Compute performance metrics."""
    with open(f'./sv/predict{model}.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(preds)
        f.close()
    accuracy=accuracy_score(labels, preds)
    precision=precision_score(labels, preds)
    recall=recall_score(labels, preds)
    f1=f1_score(labels, preds)
    # print(accuracy)
    with open('./sv/predict.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([model, accuracy, precision,recall, f1])
        file.close()
    return 0

def voting_prepare_features(texts, btext,fasttext_model, tokenizer, bert_model,a,time):
    """Prepare features by obtaining predictions from FastText and BERT models."""
    fb_time=datetime.datetime.now()
    fasttext_preds,flabel,fprobabilities = fasttext_prob(texts, fasttext_model)
    if a==1:
        compute_metrics('fasttext', flabel, label_test_list)
        plot_mix('fasttext', flabel, label_test_list)
    else:
        fprobabilities=0
    ft_end_time=datetime.datetime.now()
    
    bert_preds,blabel,bprobabilities = bert_prob(btext, tokenizer, bert_model)
    if a==1:
        compute_metrics('bert',blabel, label_test_list)
        plot_mix('bert',blabel,label_test_list)
    else:
        bprobabilities=0
    bt_end_time=datetime.datetime.now()
    
    fasttext_preds=np.array(fasttext_preds)
    bert_preds=np.array(bert_preds)
    vprobabilities=[]
    if a==1:
        # 计算预测概率的平均值
        vproba = (bert_preds + fasttext_preds) / 2
        for d in vproba:
            proba=d[1]
            vprobabilities.append(proba)
        # 确定最终预测的类别
        final_test_voting_predictions = np.argmax(vproba, axis=1)
        compute_metrics('voting',final_test_voting_predictions, label_test_list)
        plot_mix('voting',final_test_voting_predictions,label_test_list)
        print('ft_time_____:',ft_end_time-time)
        print('bt_time_____:',bt_end_time-ft_end_time+fb_time-time)
        voting_end=datetime.datetime.now()
        print('voting______:',voting_end-time)
    stack_preds=np.column_stack((fasttext_preds, bert_preds))
    return stack_preds,fprobabilities,bprobabilities,vprobabilities


def fasttext_prob(texts,fasttext_model):
    predicted_probs = []
    probabilities = []
    label=[]
    for text in texts:
        # 使用predict方法获取预测标签及其概率，k=1返回最可能的一个标签及其概率
        predicted_label, prob = fasttext_model.predict(text, k=-1)
        proba = prob[predicted_label.index('__label__1')] if '__label__1' in predicted_label else 0
        # print('------',prob)
        probabilities.append(proba)
        # 确保概率值和它的补数都被转换为浮点数
        if predicted_label[0] == '__label__1':
            a = prob[0]
            b = prob[1]
            label.append(1)
            predicted_probs.append([b,a])
        else:
            label.append(0)
            predicted_probs.append(prob)
    # print(label,predicted_probs,probabilities)
    return predicted_probs,label,probabilities

def bert_prob(texts, tokenizer, model):
    """Get prediction probabilities from BERT model."""
    model.eval()
    predicted_probs = []
    probabilities=[]
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
            predicted_probs.extend(probs)
            probs = probs[0][1]
            probabilities.append(probs)
    label = np.argmax(predicted_probs, axis=1)
    # print(label,predicted_probs,probabilities)
    return predicted_probs,label,probabilities

def train_meta_model(features, labels):
    """Train the meta model using logistic regression."""
    param_grid = [
        {
            'penalty': ['l2'],  # 'lbfgs', 'newton-cg', 'sag', 'saga' 支持这些
            'C': [0.01, 0.1, 1, 10, 100],#
            'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']#
        },
        {
            'penalty': ['l1'],  # 'saga' 和 'liblinear' 支持 l1
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['saga', 'liblinear']
        },
        {
            'penalty': ['elasticnet'],  # 仅 'saga' 支持 elasticnet
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['saga'],
            'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # 添加不同的 l1_ratio 值
        }
    ]
    meta_model = LogisticRegression(max_iter=10000)
    grid_search = GridSearchCV(estimator=meta_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1,scoring='accuracy',refit=True)
    grid_search.fit(features, labels)
    return grid_search

def plot_mix(model,val_preds,val_labels):
    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])

                # 使用pandas将混淆矩阵转换为DataFrame，以便更好地显示标签
    cm_df = pd.DataFrame(cm, index=['实际: 0', '实际: 1'], columns=['预测: 0', '预测: 1'])

                # 使用seaborn绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('实际值')
    plt.xlabel('预测值')
    plt.tight_layout(pad=1.0)  # 设置单倍行距  
    plt.savefig(f'./sv/cm_model_{model}.png')
    return 0
    
if __name__ == '__main__':
    begin_time=datetime.datetime.now()
    fasttext_model = fasttext.load_model('./fasttext/ft_model.bin')
    tokenizer = BertTokenizer.from_pretrained("./bert/bert-base-chinese")
    bert_model = BertForSequenceClassification.from_pretrained("./bert/bert_best_model")
    with open('./sv/predict.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    # 写入标题行
        writer.writerow(['model', 'Accuracy', 'precision','recall', 'f1'])


    train_data = pd.read_csv('./data/data.csv')
    msg_train_list = train_data['msg_new'].astype(str).tolist()
    bmsg_train_list = train_data['message'].astype(str).tolist()
    label_train_list = train_data['label'].tolist()
    # print(label_train_list)
    train_features,fprobabilities,bprobabilities,vprobabilities = voting_prepare_features(msg_train_list,bmsg_train_list, fasttext_model, tokenizer, bert_model,a=0,time=0)
    grid_search = train_meta_model(train_features, label_train_list)
    stacking_train_end_time=datetime.datetime.now()
    print('stacking_train_time____:',stacking_train_end_time-begin_time)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)



    # grid_search = joblib.load('./meta_model.pkl')
    joblib.dump(grid_search.best_estimator_, './sv/meta_model.pkl')
    stacking_test_begin_time=datetime.datetime.now()
    # Evaluate and log metrics
    test_data = pd.read_csv('./data/test.csv')
    msg_test_list = test_data['msg_new'].astype(str).tolist()
    bmsg_test_list = test_data['message'].astype(str).tolist()
    label_test_list = test_data['label'].tolist()
    test_features,fprobabilities,bprobabilities,vprobabilities = voting_prepare_features(msg_test_list,bmsg_test_list, fasttext_model, tokenizer, bert_model,a=1,time=stacking_test_begin_time)
    final_predictions = grid_search.predict(test_features)
    sprobabilities=[]
    final_probs = grid_search.predict_proba(test_features)
    for prob in final_probs:
        prob = prob[1]
        sprobabilities.append(prob)
    compute_metrics('stacking',final_predictions, label_test_list)
    plot_mix('stacking',final_predictions,label_test_list)
    stacking_test_end_time=datetime.datetime.now()
    print('stacking_test_time____:',stacking_test_end_time-stacking_test_begin_time)
    # print(fprobabilities,'-----',bprobabilities,'---------',sprobabilities)

    lbegin_time=datetime.datetime.now()
    logistic_model = joblib.load('./logistic/logistic_regression_best_model.pkl')
    lprobabilities=[]
    probs = logistic_model.predict_proba(msg_test_list)
    for prob in probs:
        prob = prob[1]
        lprobabilities.append(prob)
    # 计算ROC曲线的FPR和TPR
    lfpr, ltpr, _ = roc_curve(label_test_list, lprobabilities)
    # 计算AUC分数
    lroc_auc = auc(lfpr, ltpr)
    lend_time=datetime.datetime.now()
    print('逻辑回归预测时间：',lend_time-lbegin_time)
    # 计算ROC曲线的FPR和TPR
    ffpr, ftpr, _ = roc_curve(label_test_list, fprobabilities)
    # 计算AUC分数
    froc_auc = auc(ffpr, ftpr)

    # 计算ROC曲线的FPR和TPR
    bfpr, btpr, _ = roc_curve(label_test_list, bprobabilities)
    # 计算AUC分数
    broc_auc = auc(bfpr, btpr)



    # 计算元模型的ROC曲线的FPR和TPR
    sfpr, stpr, _ = roc_curve(label_test_list, sprobabilities)
    # 计算元模型的AUC分数
    sroc_auc = auc(sfpr, stpr)

    # 计算ROC曲线的FPR和TPR
    vfpr, vtpr, _ = roc_curve(label_test_list, vprobabilities)
    # 计算AUC分数
    vroc_auc = auc(vfpr, vtpr)



    # 假设你已经计算了两个模型的FPR、TPR和AUC
    fpr0, tpr0, roc_auc0 = lfpr,ltpr,lroc_auc
    fpr1, tpr1, roc_auc1 = ffpr,ftpr,froc_auc
    fpr2, tpr2, roc_auc2 =bfpr,btpr,broc_auc
    fpr3, tpr3, roc_auc3 =vfpr,vtpr,vroc_auc
    fpr4, tpr4, roc_auc4 =sfpr,stpr,sroc_auc

    with open('./sv/AUC0.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['fpr', 'tpr', 'auc'])
        writer.writerow([fpr0, tpr0, roc_auc0])
        file.close()
    with open('./sv/AUC1.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['fpr', 'tpr', 'auc'])
        writer.writerow([fpr1, tpr1, roc_auc1])
        file.close()
    with open('./sv/AUC2.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['fpr', 'tpr', 'auc'])
        writer.writerow([fpr2, tpr2, roc_auc2])
        file.close()

    with open('./sv/AUC3.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['fpr', 'tpr', 'auc'])
        writer.writerow([fpr3, tpr3, roc_auc3])
        file.close()

    with open('./sv/AUC4.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['fpr', 'tpr', 'auc'])
        writer.writerow([fpr4, tpr4, roc_auc4])
        file.close()

    plt.figure()
    lw = 2
    plt.plot(fpr0, tpr0, color='green', lw=lw, label='逻辑回归模型 (area = %0.2f)' % roc_auc0)  
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='FastText模型 (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', lw=lw, label='BERT模型 (area = %0.2f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='green', lw=lw, label='投票集成模型 (area = %0.2f)' % roc_auc3)
    plt.plot(fpr4, tpr4, color='green', lw=lw, label='堆叠集成模型 (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout(pad=1.0)  # 设置单倍行距  
    plt.legend(loc="lower right")
    plt.savefig('./sv/ROC.pdf')


    plt.figure()
    lw = 2
    plt.plot(fpr0, tpr0, color='green', lw=lw, label='逻辑回归模型 (area = %0.2f)' % roc_auc0)  
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='FastText模型 (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', lw=lw, label='BERT模型 (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout(pad=1.0)  # 设置单倍行距  
    plt.legend(loc="lower right")
    plt.savefig('./sv/ROC_dan.pdf')

    plt.figure()
    lw = 2
    plt.plot(fpr2, tpr2, color='blue', lw=lw, label='BERT模型 (area = %0.2f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='green', lw=lw, label='投票集成模型 (area = %0.2f)' % roc_auc3)
    plt.plot(fpr4, tpr4, color='green', lw=lw, label='堆叠集成模型 (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout(pad=1.0)  # 设置单倍行距  
    plt.legend(loc="lower right")
    plt.savefig('./sv/ROC_ji.pdf')