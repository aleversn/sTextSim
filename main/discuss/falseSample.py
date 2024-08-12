# %%
# this file is used to analyze the false samples in the prediction file
# and save them into different files based on their scores
# %%
import json

MAX_SCORE = 5
SAVE_DIR = './'

with open('/home/lpc/repos/sTextSim/dataset/stsbenchmark/sts-dev.json') as f:
    data = json.load(f)

with open('/home/lpc/repos/sTextSim/data_record/SimCSE_STSDANEGUn_unsup_backup1/predict_gold250.csv') as f:
    ori_data = f.readlines()

result = []
for idx, line in enumerate(ori_data):
    pred, gold = line.strip().split('\t')
    pred = float(pred)
    gold = float(gold)
    pred = pred * MAX_SCORE
    if abs(pred - gold) >= 0.2 * MAX_SCORE:
        result.append({
            'idx': idx,
            'text1': data[idx]['text1'],
            'text2': data[idx]['text2'],
            'pred': pred,
            'gold': gold
        })

FP = []
FN = []
EP = []
EN = []

for item in result:
    if item['gold'] / MAX_SCORE >= 0.5:
        if item['pred'] > item['gold']:
            EP.append(item)
        else:
            FN.append(item)
    else:
        if item['pred'] > item['gold']:
            FP.append(item)
        else:
            EN.append(item)

print(f'FP: {len(FP)} FN: {len(FN)} EP: {len(EP)} EN: {len(EN)}')

with open(SAVE_DIR + f'/FP.json', mode='w+') as f:
    json.dump(FP, f, indent=4)

with open(SAVE_DIR + f'/FN.json', mode='w+') as f:
    json.dump(FN, f, indent=4)

with open(SAVE_DIR + f'/EP.json', mode='w+') as f:
    json.dump(EP, f, indent=4)

with open(SAVE_DIR + f'/EN.json', mode='w+') as f:
    json.dump(EN, f, indent=4)

# %%
