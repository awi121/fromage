"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
import json
# from fromage import utils
import torch.nn.functional as F
import pickle
import pdb




def get_dataset(args,split, tokenizer, precision: str = 'fp32') -> Dataset:
    dataset= CsvDataset(tokenizer, '/home/ubuntu/Project/11-777-Project-aa/Data/WebQA_train_val_shruti_2.json',split=split)
    return dataset


class CsvDataset(Dataset):
  def __init__(self, tokenizer,dataset_json_path,use_num_samples=-1,
               split=['val'],Qcate=['all'], max_len: int = 32, 
               sep="\t", precision: str = 'fp32'):
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision

    assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist! {}".format(dataset_json_path)
    # self.imgid_map=pickle.load(open('/home/ubuntu/Project/11-777-Project-aa/Data/image_id_map_0328.pkl', "rb"))

    self.instance_list = []
    count = 0
    with open(dataset_json_path, "r") as f:
        dataset_J = json.load(f)
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if datum['img_posFacts_path'] is not None:
                  if use_num_samples == -1 or count < use_num_samples:
                      guid = datum['Guid']
                      qcate = datum['Qcate'] if 'Qcate' in datum else 'TBD'
                      Q = datum['Q'].replace('"', "")
                      A = datum['A'][0].replace('"', "")
                      A_list = [a.replace('"', "") for a in datum['A']]
                      try: Keywords_A = datum['Keywords_A'].replace('"', "")
                      except: Keywords_A = "TBD"
                      gold_facts = []
                      
    
                      for idx,fa in enumerate(datum['txt_posFacts_path']):
                          # encode fa using a encoder function and save the path by creating a new key 'txt_posFacts_path'
                          fa=fa.replace('/Users/aavi/Desktop/11-777-Project/Data/univlr_features','/home/ubuntu/Project/11-777-Project-aa/Data/univlr_features')
                          gold_facts.append(fa)
                      self.instance_list.append((gold_facts, [], [], [], Q, A, Keywords_A, A_list, False, "txt", guid, qcate)) # do_filter_task, context
                      
                      count += 1
                else: 
                  if ('all' in Qcate) or datum['Qcate'] in Qcate:
                      if use_num_samples == -1 or count < use_num_samples:
                          guid = datum['Guid']
                          qcate = datum['Qcate'] if 'Qcate' in datum else 'TBD'
                          Q = datum['Q'].replace('"', "")
                          A = datum['A'][0].replace('"', "")
                          A_list = [a.replace('"', "") for a in datum['A']]
                          try: Keywords_A = datum['Keywords_A'].replace('"', "")
                          except: Keywords_A = "TBD"
                          gold_feature = []
            
                          for idx,fa in enumerate(datum['img_posFacts_path']):
                          # encode fa using a encoder function and save the path by creating a new key 'txt_posFacts_path'
                            fa=fa.replace('/Users/aavi/Desktop/11-777-Project/Data/univlr_features','/home/ubuntu/Project/11-777-Project-aa/Data/univlr_features')
                            gold_feature.append(fa)
                          self.instance_list.append((gold_feature, [], [], [], Q, A, Keywords_A, A_list, False, "img", guid, qcate)) # do_filter_task, context )
                          count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))



    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.instance_list)

  def __getitem__(self, idx):
    gold_feature, [], [], [], Q, A, Keywords_A, A_list, _, type1, _, qcate= self.instance_list[idx]
    while(len(gold_feature)<2):
      gold_feature.append(gold_feature[0])
    tokenized_data = self.tokenizer(
        A,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=self.max_len)
    tokens = tokenized_data.input_ids[0]
    Q='Q: ' + Q + '\nA:'
    caption_len = tokenized_data.attention_mask[0].sum()

      
    return gold_feature, Q, tokens, caption_len,type1
  