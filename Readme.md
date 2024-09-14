
### `SimCSE`训练
- model: BERT/RoBERTa

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from main.trainers.cse_trainer import Trainer
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('/home/lpc/models/bert-base-uncased')
trainer = Trainer(tokenizer=tokenizer,
                  from_pretrained='/home/lpc/models/esimcse-bert-base-uncased/',
                  data_present_path='./dataset/present.json',
                  max_seq_len=32,
                  hard_negative_weight=0,
                  batch_size=64,
                  temp=0.05,
                  data_name='STSDASimple',
                  task_name='ESimCSE_STSDASimple_unsup')

for i in trainer(num_epochs=3, lr=2e-5, gpu=[0], eval_call_step=lambda x: x % 125 == 0, save_per_call=True):
    a = i
```

```python
# 如果要单独验证请注释掉上述训练的for循环，然后运行下面的代码

trainer.eval(0, is_eval=True)
```

### 使用`SentEval`评估模型性能(需要回退Numpy到1.23.5)
 
支持通过三种方式执行评估：

- 1. 通过bash脚本执行, 其中`tokenizer_path`缺省时默认使用`model_name_or_path`指定的模型的tokenizer

```bash
!python main/evaluation.py --model_name_or_path='./model/bert-base-uncased' --tokenizer_path='./model/bert-base-uncased'
```

- 2. 通过`ipynb`执行, 其中在默认情况下将会根据`model_name_or_path`自动选择模型类型

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from main.evaluation import *

model_path = '/home/lpc/models/unsup-simcse-bert-base-uncased/'
tokenizer_path = '/home/lpc/models/unsup-simcse-bert-base-uncased/'

main([
    '--model_name_or_path', model_path,
    '--tokenizer_path', tokenizer_path,
    '--task_set', 'transfer'
])
```

- 3. 也可以通过自定义`model`传入到`main()`函数来执行评估, 若采用本项目中定义的模型建议采用以下方式执行评估, 否则可能会出现丢失部分模型参数的情况

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from main.evaluation import *
from main.models.gcse import GCSE

model_path = '/home/lpc/repos/sTextSim/save_model/G-ESimCSELarge_TransferDAFullUn_unsup_sota/GCSE_step_750'
tokenizer_path = '/home/lpc/models/esimcse-bert-large-uncased/'
model = GCSE(from_pretrained=model_path,
                                pooler_type='cls')

main([
    '--model_name_or_path', model_path,
    '--tokenizer_path', tokenizer_path,
    '--task_set', 'transfer'
], model)
```

### 查看保存模型的权重

```python
import torch

# 指定预训练模型的路径
model_path = 'save_model/SimCSE_Wiki_unsup/simcse_best/pytorch_model.bin'

# 使用 torch.load 加载模型
state_dict = torch.load(model_path, map_location='cpu')

# 打印模型的键（参数的名称）
print("Model Keys:", state_dict.keys())

# 打印模型的详细信息
for key, value in state_dict.items():
    print(f"Key: {key}, Shape: {value.shape}")
```

### `NLI`训练
- model: BERT/SBERT

```python
from main.trainers.sts_trainer import Trainer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./model/chinese_wwm_ext')
trainer = Trainer(tokenizer=tokenizer,
                  from_pretrained='./model/chinese_wwm_ext',
                  model_type='bert',
                  data_present_path='./dataset/present.json',
                  max_seq_len=128,
                  hard_negative_weight=0,
                  batch_size=64,
                  temp=0.05,
                  data_name='CNSTS',
                  task_name='BERT_CNSTS')

for i in trainer(num_epochs=15, lr=5e-5, gpu=[0], eval_call_step=lambda x: x % 250 == 0):
    a = i
```

```python
# 如果要单独验证请注释掉上述训练的for循环，然后运行下面的代码

trainer.eval(0, is_eval=True)
```

