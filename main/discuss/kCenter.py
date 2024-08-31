# %%
# Using K-center to filter the most uniform subset from generated dataset.
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

def get_embeddings(model, batch, pooler='cls', gpu=[0]):
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpu).cuda()
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return pooler_output.cpu()
    elif pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(
            1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                         batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    else:
        raise NotImplementedError

def batcher(model, tokenizer, input_list, batch_size=128, gpu=[0]):

    num_batch = len(input_list) // batch_size + 1

    embeddings_list = []
    for i in tqdm(range(num_batch)):
        batch = input_list[i*batch_size:(i+1)*batch_size]
        batch = tokenizer(batch, padding=True,
                          truncation=True, return_tensors='pt')
        embeddings = get_embeddings(model, batch, gpu=gpu)
        embeddings_list += embeddings.tolist()
    return embeddings_list

# 计算list1和list2的余弦距离
def get_distance(list1, list2):
    list1 = torch.tensor(list1).cuda()
    list2 = torch.tensor(list2).cuda()
    list1_norm = list1 / list1.norm(dim=1, keepdim=True)
    list2_norm = list2 / list2.norm(dim=1, keepdim=True)

    cosine_distances = 1 - torch.mm(list1_norm, list2_norm.T)
    return cosine_distances

def k_center_greedy(cosine_distances, selected_indexes=[], buget=10000):
    cos = cosine_distances.clone().T
    cos[selected_indexes] = -torch.inf
    min_dis_val, min_dis_index = cos.min(dim=1)
    sort_dis_value, sort_dis_index = torch.sort(min_dis_val, dim=0, descending=True)
    sort_cor_dis_index = min_dis_index[sort_dis_index]
    return {
        'target_selected_dis': sort_dis_value.tolist()[:buget],
        'target_selected_index': sort_dis_index.tolist()[:buget],
        'target_selected_cor_index': sort_cor_dis_index.tolist()[:buget]
    }

def r_k_center(target_selected_obj, error_num=500):
    target_selected_dis = target_selected_obj['target_selected_dis']
    target_selected_index = target_selected_obj['target_selected_index']
    target_selected_cor_index = target_selected_obj['target_selected_cor_index']

    target_selected_dis, sort_indexes = torch.sort(torch.tensor(target_selected_dis), dim=0, descending=True)
    target_selected_cor_index = torch.tensor(target_selected_cor_index)[sort_indexes]
    target_selected_index = torch.tensor(target_selected_index)[sort_indexes]
    phi = torch.max(torch.tensor(target_selected_dis)).tolist()
    lb = phi / 2
    ub = phi
    while round(lb, 12) != round(ub, 12):
        target_error_num = 0
        mid = (lb + ub) / 2
        for idx, target_dis in enumerate(target_selected_dis):
            if target_dis > mid:
                target_error_num += 1
        if target_error_num > error_num:
            lb = mid
        else:
            ub = mid
    
    final_phi = (lb + ub) / 2
    result = []
    for idx, target_dis in enumerate(target_selected_dis):
        if target_dis <= final_phi:
            result.append((target_selected_index[idx].tolist(), target_selected_cor_index[idx].tolist(), target_selected_dis[idx].tolist()))
    return result, final_phi


from_pretrained_path = '/home/lpc/models/sup-simcse-bert-base-uncased/'
tokenizer = BertTokenizer.from_pretrained(from_pretrained_path)
model = BertModel.from_pretrained(from_pretrained_path)

with open('/home/lpc/repos/sTextSim/dataset/stsbenchmark/sts-train.json') as f:
    ori_json = json.load(f)

source_list = []
for item in ori_json:
    source_list.append(item['text1'])
    source_list.append(item['text2'])

emb_sts = batcher(model, tokenizer, source_list, batch_size=512, gpu=[0, 1, 2, 3, 4, 5, 6, 7])

target_list = []

with open('/home/lpc/repos/sTextSim/dataset/wiki_DA/full_generated.jsonl') as f:
    ori_json = f.readlines()

for item in ori_json:
    item = json.loads(item)
    target_list.append(item['text1'])

emb_wiki = batcher(model, tokenizer, target_list, batch_size=512, gpu=[0, 1, 2, 3, 4, 5, 6, 7])

cos_socres = get_distance(emb_sts, emb_wiki)


# %%
# (0.9625331163406372, 1.4541891813278198, -1.430511474609375e-06)
target_selected_obj = k_center_greedy(cos_socres, buget=50000)

# %%
result, phi = r_k_center(target_selected_obj)

# %%
with open('/home/lpc/repos/sTextSim/dataset/wiki_DA/k_center_generated.jsonl', mode='w+') as f:
    for item in tqdm(result):
        json_item = ori_json[item[0]]
        f.write(json_item)

# %%
