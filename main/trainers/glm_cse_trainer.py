import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numpy as np
from tqdm import tqdm
from main.analysis import Analysis
from torch.utils.data import DataLoader
from main.loaders.glm_cl_loader import CLDataset
from transformers import AutoConfig, PreTrainedModel
from main.models.chatglm_lora import ChatGLMLoRACSE
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType
from peft.config import PeftConfig
from peft.utils import _prepare_prompt_learning_config
from accelerate import Accelerator

accelerator = Accelerator()

def get_peft_model(
    model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", autocast_adapter_dtype: bool = True,
    revision = None, pooler_type='cls', hard_negative_weight=0, temp=0.05):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return ChatGLMLoRACSE(model, peft_config, adapter_name=adapter_name, pooler_type=pooler_type, hard_negative_weight=hard_negative_weight, temp=temp, autocast_adapter_dtype=autocast_adapter_dtype)


class Trainer():
    model: ChatGLMLoRACSE
    def __init__(
        self, tokenizer, from_pretrained=None, resume_path=None, autocast_adapter_dtype = True, data_name='default', data_present_path=None, train_file=None, eval_file=None, test_file=None, max_seq_len=256, batch_size=2, batch_size_eval=32, eval_label_scale=5.0, hard_negative_weight=0, temp=0.05, eval_mode='dev', task_name='SimCSE'
    ):

        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.resume_path = resume_path
        self.autocast_adapter_dtype = autocast_adapter_dtype
        self.data_name = data_name
        self.data_present_path = data_present_path
        self.accelerate = accelerator
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.eval_label_scale = eval_label_scale
        self.hard_negative_weight = hard_negative_weight
        self.temp = temp
        self.eval_mode = eval_mode

        self.dataloader_init()
        self.model_init()
        self.analysis = Analysis()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained, trust_remote_code=True)
        if self.config.model_type == 'chatglm':
            self.model = AutoModel.from_pretrained(
                self.from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.1,
            )
            if self.resume_path is not None:
                print('Accessing Resume PATH: {} ...\n'.format(self.resume_path))
                self.model.enable_input_require_grads()
                self.model = ChatGLMLoRACSE.from_pretrained(
                    self.model, self.resume_path, config=peft_config)
            else:
                self.model = get_peft_model(self.model, peft_config, pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp)

    def dataloader_init(self):
        if self.data_present_path is None:
            self.data_path = {
                'train': self.train_file,
                'dev': self.eval_file,
                'test': self.test_file
            }
        else:
            self.data_path = self.get_data_present(
                self.data_present_path)[self.data_name]
        self.train_set = CLDataset(
            self.tokenizer, self.data_path['train'], max_seq_len=self.max_seq_len, shuffle=True)
        self.eval_set = CLDataset(
            self.tokenizer, self.data_path['dev'], max_seq_len=self.max_seq_len, shuffle=False, is_eval=True)
        if 'test' in self.data_path and self.data_path['test'] is not None:
            self.test_set = CLDataset(
                self.tokenizer, self.data_path['test'], max_seq_len=self.max_seq_len, shuffle=False, is_eval=True)
        self.train_loader = DataLoader(
            self.train_set, self.batch_size)
        self.eval_loader = DataLoader(
            self.eval_set, self.batch_size_eval) if self.eval_mode == 'dev' else DataLoader(self.test_set, self.batch_size_eval)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def __call__(self, resume_step=None, num_epochs=30, lr=5e-5, eval_call_step=None):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_step=eval_call_step)

    def train(self, resume_step=None, num_epochs=30, lr=5e-5, eval_call_step=None):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        self.model, optimizer, train_loader, scheduler = self.accelerate.prepare(self.model, optimizer, self.train_loader, scheduler)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        best_eval_score = 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0

            train_iter = tqdm(train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                loss = outputs.loss
                loss = loss.mean()

                # loss.backward()
                self.accelerate.backward(loss)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count)

                if (eval_call_step is None and train_step % 125 == 0) or eval_call_step(train_step):
                    X, Y, score = self.eval(train_step)
                    if score > best_eval_score:
                        best_eval_score = score
                        self.save_model('best')
                    self.analysis.save_pred_gold(
                        X, Y, uid=current_uid if self.task_name is None else self.task_name, step=train_step)
                    self.analysis.save_all_records(
                        uid=current_uid if self.task_name is None else self.task_name)
                    yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, 'current_best')
                    self.model.train()

            model_uid = self.save_model(epoch + 1)

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count
            })

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)
    
    @accelerator.on_local_main_process
    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists(f'./save_model/{dir}'):
            os.makedirs(f'./save_model/{dir}')
        save_model = self.accelerate.unwrap_model(self.model)
        save_model.model.save_pretrained(
            f'./save_model/{dir}/simcse_{current_step}',
            is_main_process=self.accelerate.is_main_process,
            save_function=self.accelerate.save,
        )
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, is_eval=False, pure_eval=False):
        if pure_eval:
            self.model = self.accelerate.prepare_model(self.model)
        
        self.eval_loader = self.accelerate.prepare_data_loader(self.eval_loader)

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            X = []
            Y = []
            eval_spearman = []

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                loss = outputs.loss
                logits = outputs['logits']
                loss = loss.mean()

                p = torch.diag(logits) * self.temp
                # print(logits[29] * self.temp)
                # text1 = self.tokenizer.decode(it['input_ids'][29][0], skip_special_tokens=True)
                # text2 = self.tokenizer.decode(it['input_ids'][29][1], skip_special_tokens=True)
                # print(text1, text2)
                # print(0 / 0)

                eval_loss += loss.data.item()
                eval_count += 1

                gold = it['labels']
                X += p.tolist()
                Y += gold.tolist()
                tp += ((gold / self.eval_label_scale >= 0.5) & (p >= 0.5)).sum().item()
                fp += ((gold / self.eval_label_scale < 0.5) & (p >= 0.5)).sum().item()
                fn += ((gold / self.eval_label_scale >= 0.5) & (p < 0.5)).sum().item()
                tn += ((gold / self.eval_label_scale < 0.5) & (p < 0.5)).sum().item()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                eval_spearman.append(spearmanr)

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, Spearman=np.mean(spearmanr), eval_acc=(tp + tn) / (tp + tn + fp + fn), precision=precision, recall=recall, f1=f1)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                'eval_acc': (tp + tn) / (tp + tn + fp + fn),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'spearman': np.mean(eval_spearman)
            })

        if is_eval:
            self.analysis.save_pred_gold(
                X, Y, uid='0' if self.task_name is None else self.task_name, step=0)
            print(f'F1: {f1}, Precision: {precision}, Recall: {recall}')

        return X, Y, np.mean(eval_spearman)

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX
