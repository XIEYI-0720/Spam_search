import re
import jieba
import pandas as pd


def find_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese_txt = re.sub(pattern, '', text)
    return chinese_txt
def preprocess_text(data):
    with open('./stop_words.txt', 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file]
    data['msg_new'] = data['message'].apply(find_chinese)
    data_shuffled = data.sample(frac=1, random_state=11).reset_index(drop=True)  # 混洗
    for index, row in data_shuffled.iterrows():
        text = row['msg_new']
        tokens = jieba.lcut(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]  # 过滤掉停止词
        data_shuffled.at[index, 'msg_new'] = ' '.join(filtered_tokens)
        filtered_text = ' '.join(filtered_tokens)  # 用空格连接过滤后的单词
        
        data_shuffled.at[index, 'msg_new'] = filtered_text
    return data_shuffled

def choose(df, x, y):
    label_0_data = df[df['label'] == 0]
    label_1_data = df[df['label'] == 1]
    # 从每个标签对应的数据中抽取记录
    # print(len(label_0_data))
    sampled_label_0_data = label_0_data.sample(n=x, random_state=111)
    sampled_label_1_data = label_1_data.sample(n=y, random_state=110)
    # 合并抽样后的数据
    csvfile = pd.concat([sampled_label_0_data, sampled_label_1_data])

    data_shuffled0 = csvfile.sample(frac=1, random_state=11).reset_index(drop=True)
    # 从原始数据中移除已抽取的记录
    remaining_label_0_data = label_0_data.drop(sampled_label_0_data.index)
    remaining_label_1_data = label_1_data.drop(sampled_label_1_data.index)
# 保存未抽取的记录到另一个文件
    csvfile_non = pd.concat([remaining_label_0_data, remaining_label_1_data])
    data_shuffled = csvfile_non.sample(frac=1, random_state=11).reset_index(drop=True)
    return data_shuffled0, data_shuffled


if __name__ == '__main__':
    data = pd.read_csv('./pre_data.csv')
    # del data['msg_new']
    # print(data.isnull().sum())
    # data = preprocess_text(data)
    # data = data[data['msg_new'].str.strip().astype(bool)]
    # data.to_csv('../data/all_data.csv', index=False, encoding='utf-8')
    # data = data.dropna()


    file, file_non=choose(data, x=80000, y=57000)
    file.to_csv('./pre_data.csv', index=False, encoding='utf-8')

    # file_non.to_csv("../data/data_non.csv", index=False, encoding='utf-8')

    # df = pd.read_csv('./use_data.csv')
    # csvfile, csvfile_non=choose(df, x=10000, y=8000)
    # csvfile.to_csv('./data.csv', index=False, encoding='utf-8')
    # csvfile_non.to_csv("./test.csv", index=False, encoding='utf-8')

    print('okkk')
