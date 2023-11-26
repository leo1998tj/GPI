from pandarallel import pandarallel
import pandas as pd
import numpy as np
import json
import os
import six
from collections import Counter
from docx import Document
import os
import re
import pdb
import re
from random import choice
import random
import torch
from transformers import AutoTokenizer, BertModel
from datasets import Dataset
import torch
from tqdm import tqdm
torch.cuda.is_available()
pandarallel.initialize()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
pretrained = BertModel.from_pretrained("bert-base-chinese")
model_for_test = Model()
model_for_test.load_state_dict(torch.load('models/model_state_dict_4epoch_4400iter.pkl'))
dataset_val = pd.read_csv("./0414res/test_dataset/test_dataset.csv", index_col=0)
device = torch.device("cpu")	
model_for_test = model_for_test.to(device)
pretrained = pretrained.to(device)

def get_lable_res(test_text):
    test_text = test_text[:510] # tokenizer 不提供截断padding接口，人为截断
    inputs = tokenizer(test_text, return_tensors="pt")
    res = model_for_test(input_ids=inputs["input_ids"].to(device), 
                        attention_mask=inputs["attention_mask"].to(device), 
                        token_type_ids=inputs["token_type_ids"].to(device))
    return res.cpu().detach().numpy()[0][0], res.cpu().detach().numpy()[0][1], res.argmax(dim=1).cpu().detach().numpy()[0]


dataset_val_1 = dataset_val.head(1000)
tqdm.pandas(desc="get feature in input_data")
# dataset_val_1["neg_prob"], dataset_val_1["pos_prob"], dataset_val_1["lable"] = zip(*dataset_val_1.parallel_apply(lambda x: get_lable_res(x.text), axis=1))
dataset_val_1["neg_prob"], dataset_val_1["pos_prob"], dataset_val_1["lable"] = zip(*dataset_val_1.apply(lambda x: get_lable_res(x.text), axis=1))
# dataset_val["neg_prob"], dataset_val["pos_prob"], dataset_val["lable"] = zip(*dataset_val.parallel_apply(lambda x: get_lable_res(x.text), axis=1))
# dataset_val.to_csv("0414res/trained_res.csv", encoding="utf-8")
dataset_val_1.to_csv("0414res/trained_res.csv", encoding="utf-8")