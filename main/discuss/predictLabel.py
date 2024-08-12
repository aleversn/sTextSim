# %%
# 使用SimCSE Baseline预测生成数据集的相关性

import os
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from main.predictors.cse_predictor import Predictor
from transformers import BertTokenizer

with open('./dataset/sts_DA/full_generated_neg.json') as f:
    data = json.load(f)

input_list = []
for item in data:
    input_list.append([item['text1'], item['text2']])
    input_list.append([item['text1'], item['neg']])

tokenizer = BertTokenizer.from_pretrained('/home/lpc/models/bert-base-uncased')
pred = Predictor(tokenizer=tokenizer,
                  from_pretrained='/home/lpc/models/unsup-simcse-bert-base-uncased/',
                  max_seq_len=32,
                  hard_negative_weight=0,
                  batch_size=64,
                  temp=0.05)

logits = []
for output in pred.pred(input_list):
    logits += output.logits.diag().tolist()

# %%
for idx, item in enumerate(tqdm(data)):
    pos_label = logits[idx * 2]
    neg_label = logits[idx * 2 + 1]
    item['pos_label'] = pos_label
    item['neg_label'] = neg_label

# %%
with open('./dataset/sts_DA/full_generated_neg_with_label.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# %%
# 计算数据集相关性的分数正态分布, 并选取阈值超参数

import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde

# 读取样本数据
with open('/home/lpc/repos/sTextSim/dataset/sts_DA/full_generated_neg_with_label_sup.json') as f:
    json_data = json.load(f)
temp = 1
data = []
for item in json_data:
    data.append(float(item['pos_label']) * temp)
    data.append(float(item['neg_label']) * temp)
data = np.array(data)

# 使用核密度估计拟合数据分布
kde = gaussian_kde(data)

# 生成 x 轴的值范围
x = np.linspace(min(data), max(data), 1000)
pdf = kde(x)

# 计算密度估计的累积分布函数 (CDF)
cdf = np.cumsum(pdf) / np.sum(pdf)

# 找到累积密度为 10% 和 90% 的位置
percentile_10 = np.interp(0.1, cdf, x)
percentile_90 = np.interp(0.9, cdf, x)

print(f'Position value at 10% density: {percentile_10}')
print(f'Position value at 90% density: {percentile_90}')

# 绘制直方图和核密度估计曲线
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Sample Data')
plt.plot(x, pdf, 'r', linewidth=2, label='KDE Fitted Distribution')
plt.title('Kernel Density Estimation (KDE) Fitting')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# %%
