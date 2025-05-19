import os
import torch
import torch.functional as F
import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr


class Analysis():

    def __init__(self):
        self.train_record = {}
        self.eval_record = {}
        self.model_record = {}

    '''
    append data record of train
    train_record_item: dict
    '''

    def append_train_record(self, train_record_item):
        for key in train_record_item:
            if key not in self.train_record:
                self.train_record[key] = []
            self.train_record[key].append(train_record_item[key])

    '''
    append data record of eval
    eval_record_item: dict
    '''

    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            if key not in self.eval_record:
                self.eval_record[key] = []
            self.eval_record[key].append(eval_record_item[key])

    '''
    append data record of model
    uid: model uid
    '''

    def append_model_record(self, uid):
        key = "model_uid"
        if key not in self.model_record:
            self.model_record[key] = []
        self.model_record[key].append(uid)

    def save_all_records(self, uid):
        self.save_record('train_record', uid)
        self.save_record('eval_record', uid)
        self.save_record('model_record', uid)

    def save_record(self, record_name, uid):
        record_dict = getattr(self, record_name)
        path = f'./data_record/{uid}'
        if not os.path.exists(path):
            os.makedirs(path)
        head = []
        for key in record_dict:
            head.append(key)
        if len(head) == 0:
            return uid
        result = ''
        for idx in range(len(record_dict[head[0]])):
            for key in head:
                result += str(record_dict[key][idx]) + '\t'
            result += '\n'

        result = "\t".join(head) + '\n' + result

        with open(f'{path}/{record_name}.csv', encoding='utf-8', mode='w+') as f:
            f.write(result)

        return uid
    
    @staticmethod
    def computed_batch_label(logits: torch.FloatTensor, mode='regression'):
        if mode == 'regression' and logits.size(1) == 2:
            return logits[:, 1]
        else:
            return logits.argmax(dim=1)
    
    @staticmethod
    def computed_f1(pred: list, gold: list, mode='regression', num_labels=1):
        pred = torch.tensor(pred)
        gold = torch.tensor(gold)
        if mode == 'regression':
            tp = ((pred >= 0.5) & (gold >= 0.5)).sum().item()
            fp = ((pred >= 0.5) & (gold < 0.5)).sum().item()
            fn = ((pred < 0.5) & (gold >= 0.5)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            tp = {}
            fp = {}
            fn = {}
            for p_item, g_item in zip(pred.tolist(), gold.tolist()):
                if p_item == g_item:
                    tp[g_item] += 1
                else:
                    fn[g_item] += 1
                    fp[p_item] += 1
            
            result_dict = {}
            for i in range(num_labels):
                result_dict[str(i)] = {
                    'precision': tp[i] / (tp[i] + fp[i]),
                    'recall': tp[i] / (tp[i] + fn[i]),
                    'f1': 2 * tp[i] / (2*tp[i] + fp[i] + fn[i]) if tp[i] != 0 else 0,
                }
            result_dict['f1'] = np.mean([result_dict[str(i)]['f1'] for k in range(num_labels)])
            result_dict['precision'] = np.mean([result_dict[str(i)]['precision'] for k in range(num_labels)])
            result_dict['recall'] = np.mean([result_dict[str(i)]['recall'] for k in range(num_labels)])
            return result_dict

    '''
    X: the gold labels
    Y: the predicted labels
    '''
    @staticmethod
    def evaluationSAS(X, Y):
        if len(X) == 0:
            return 0, 0
        if len(X) != len(Y):
            raise Exception('mismatch length of X and Y')
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        r_a = 0
        r_b = 0
        r_c = 0
        r_mse = 0
        for i in range(len(X)):
            _x = (X[i] - x_mean)
            _y = (Y[i] - y_mean)
            r_a += _x * _y
            r_b += _x ** 2
            r_c += _y ** 2
            r_mse += (X[i] - Y[i]) ** 2
        r = r_a / (r_b ** 0.5 * r_c ** 0.5)
        r_mse = (r_mse / len(X)) ** 0.5

        return r, r_mse, pearsonr(X, Y)[0], spearmanr(X, Y)[0]

    @staticmethod
    def heatmap(data):
        return ValueError('')

    @staticmethod
    def save_pred_gold(X, Y, uid, step=None):
        path = f'./data_record/{uid}'
        if not os.path.isdir(path):
            os.makedirs(path)
        result = ''
        for i in range(len(X)):
            result += '{}\t{}\n'.format(X[i], Y[i])
        with open('{}/predict_gold{}.csv'.format(path, '' if step is None else step), encoding='utf-8', mode='w+') as f:
            f.write(result)

    @staticmethod
    def save_same_row_list(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]
                              ) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
