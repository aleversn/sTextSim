# %%
import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class CLDataset(Dataset):
    def __init__(self, tokenizer, file_name, max_seq_len=256, shuffle=True, is_eval=False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_eval = is_eval
        self.ori_data, self.compute_data = self.load_train(
            file_name)
        self.random_list = list(range(len(self.compute_data)))
        if shuffle:
            random.shuffle(self.random_list)

    def load_train(self, file_name):
        is_json = 'txt'
        if file_name.endswith('json'):
            is_json = 'json'
        if file_name.endswith('jsonl'):
            is_json = 'jsonl'
        if is_json == 'json':
            with open(file_name, encoding='utf-8') as f:
                ori_data = json.load(f)
        elif is_json == 'jsonl':
            ori_data = []
            with open(file_name, encoding='utf-8') as f:
                for line in f:
                    ori_data.append(json.loads(line))
        else:
            with open(file_name, encoding='utf-8') as f:
                _ori_data = f.read().split('\n')
            if _ori_data[-1] == '':
                _ori_data = _ori_data[:-1]
            
            ori_data = []
            for line in _ori_data:
                line = line.split('\t')
                l = {}
                l = {
                    'text1': line[0]
                }
                if len(line) > 1:
                    l['text2'] = line[1]
                if len(line) > 2:
                    l['label'] = line[2]
                ori_data.append(l)
        
        result = []
        if not self.is_eval:
            for line in ori_data:
                if 'label' in line:
                    if line['label'] == '0':
                        line['neg'] = line['text2']
                        line['text2'] = line['text1']
                    else:
                        while True:
                            sample = random.choice(ori_data)
                            if sample['text1'] != line['text1']:
                                break
                        line['neg'] = sample['text1'] 

                result.append(line)

            return ori_data, result
        
        for line in ori_data:
            result.append(line)

        return ori_data, result
    
    def get_train_item(self, idx):
        item = self.compute_data[self.random_list[idx]]
        s1 = item['text1']
        T1 = self.tokenizer(s1, add_special_tokens=True,
                            max_length=self.max_seq_len, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        if 'token_type_ids' in T1:
            tid1 = torch.tensor(T1['token_type_ids'])
        else:
            tid1 = torch.tensor([0] * ss1.size(-1))

        if 'label' in item:
            label = torch.tensor(float(item['label']))
        else:
            label = torch.tensor(1.0)

        if 'neg' in item:
            if 'text2' not in item:
                ss2 = torch.tensor(T1['input_ids'])
                mask2 = torch.tensor(T1['attention_mask'])
                if 'token_type_ids' in T2:
                    tid2 = torch.tensor(T2['token_type_ids'])
                else:
                    tid2 = torch.tensor([0] * ss2.size(-1))
            else:
                s2 = item['text2']

                T2 = self.tokenizer(s2, add_special_tokens=True,
                                    max_length=self.max_seq_len, padding='max_length', truncation=True)

                ss2 = torch.tensor(T2['input_ids'])
                mask2 = torch.tensor(T2['attention_mask'])
                tid2 = torch.ones(ss2.shape).long()

            neg = item['neg']
            T3 = self.tokenizer(neg, add_special_tokens=True,
                                max_length=self.max_seq_len, padding='max_length', truncation=True)
            
            ss3 = torch.tensor(T3['input_ids'])
            mask3 = torch.tensor(T3['attention_mask'])
            tid3 = torch.ones(ss3.shape).long()

            return {
                'input_ids': torch.stack([ss1, ss2, ss3]),
                'attention_mask': torch.stack([mask1, mask2, mask3]),
                'token_type_ids': torch.stack([tid1, tid2, tid3]),
                'labels': label
            }
        
        return {
            'input_ids': torch.stack([ss1, ss1]),
            'attention_mask': torch.stack([mask1, mask1]),
            'token_type_ids': torch.stack([tid1, tid1]),
            'labels': label
        }
    
    def get_eval_item(self, idx):
        item = self.compute_data[self.random_list[idx]]
        s1 = item['text1']
        T1 = self.tokenizer(s1, add_special_tokens=True,
                            max_length=self.max_seq_len, padding='max_length', truncation=True)
        ss1 = torch.tensor(T1['input_ids'])
        mask1 = torch.tensor(T1['attention_mask'])
        if 'token_type_ids' in T1:
            tid1 = torch.tensor(T1['token_type_ids'])
        else:
            tid1 = torch.tensor([0] * ss1.size(-1))

        s2 = item['text2']

        T2 = self.tokenizer(s2, add_special_tokens=True,
                            max_length=self.max_seq_len, padding='max_length', truncation=True)

        ss2 = torch.tensor(T2['input_ids'])
        mask2 = torch.tensor(T2['attention_mask'])
        if 'token_type_ids' in T2:
            tid2 = torch.tensor(T2['token_type_ids'])
        else:
            tid2 = torch.tensor([0] * ss2.size(-1))

        if 'label' in item:
            label = torch.tensor(float(item['label']))
        else:
            label = torch.tensor(1.0)
        
        return {
            'input_ids': torch.stack([ss1, ss2]),
            'attention_mask': torch.stack([mask1, mask2]),
            'token_type_ids': torch.stack([tid1, tid2]),
            'labels': label
        }

    def __getitem__(self, idx):
        if self.is_eval:
            return self.get_eval_item(idx)
        return self.get_train_item(idx)

    def __len__(self):
        return len(self.compute_data) // 2 * 2
