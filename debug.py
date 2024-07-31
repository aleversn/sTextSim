# %%

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 7"
from main.trainers.glm_cse_trainer import Trainer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/lpc/models/chatglm3-6b', trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer,
                  from_pretrained='/home/lpc/models/chatglm3-6b',
                  data_present_path='./dataset/present.json',
                  max_seq_len=32,
                  hard_negative_weight=0,
                  batch_size=16,
                  temp=0.05,
                  data_name='WikiSTS',
                  task_name='SimCSE_glm_wiki')

for i in trainer(num_epochs=15, lr=2e-5, eval_call_step=lambda x: x % 250 == 0):
    a = i

# %%
