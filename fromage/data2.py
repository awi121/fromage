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
import clip
import torch.nn.functional as F
import pickle
import pdb




def get_dataset(args,split, tokenizer, precision: str = 'fp32') -> Dataset:
    dataset= CsvDataset(tokenizer, '/home/ubuntu/Project/11-777-Project-aa/Data/WebQA_train_val.json',split=split)
    return dataset


class CsvDataset(Dataset):
  def __init__(self, tokenizer,dataset_json_path,use_num_samples=-1,
               split=['val'],Qcate=['all'], max_len: int = 32, 
               sep="\t", precision: str = 'fp32'):
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision

    assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist! {}".format(dataset_json_path)
    self.imgid_map=pickle.load(open('/home/ubuntu/Project/11-777-Project-aa/Data/image_id_map_0328.pkl', "rb"))

    self.instance_list = []
    count = 0
    with open(dataset_json_path, "r") as f:
        dataset_J = json.load(f)
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if datum['Qcate'] == 'text':
                   continue
                  
                  # if use_num_samples == -1 or count < use_num_samples:
                  #     guid = datum['Guid']
                  #     qcate = datum['Qcate'] if 'Qcate' in datum else 'TBD'
                  #     Q = datum['Q'].replace('"', "")
                  #     A = datum['A'][0].replace('"', "")
                  #     A_list = [a.replace('"', "") for a in datum['A']]
                  #     try: Keywords_A = datum['Keywords_A'].replace('"', "")
                  #     except: Keywords_A = "TBD"
                  #     gold_facts = []
                      
                  #     for fa in datum['txt_posFacts']:
                  #         gold_facts.append(fa['fact'])
                  #     self.instance_list.append((gold_facts, [], [], [], Q, A, Keywords_A, A_list, False, "txt", guid, qcate)) # do_filter_task, context
                      
                  #     count += 1
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
                          # pdb.set_trace()
                          if(len(datum['img_posFacts'])!=2):
                            continue
                          for im in datum['img_posFacts']:
                              image_id = int(im['image_id'])
                              image_id = self.imgid_map[image_id]
                              if os.path.exists(os.path.join("/home/ubuntu/Project/11-777-Project-aa/Data/", "{}/{}/{}.pkl".format('dev', (image_id%10000000)//1000, image_id))):
                                image_feature_path =os.path.join("/home/ubuntu/Project/11-777-Project-aa/Data/", "{}/{}/{}.pkl".format('dev', (image_id%10000000)//1000, image_id))
                                gold_feature.append(image_feature_path)
                          if gold_feature == []:
                              continue
                          self.instance_list.append((gold_feature, [], [], [], Q, A, Keywords_A, A_list, False, "img", guid, qcate)) # do_filter_task, context )
                          count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))



    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.instance_list)

  def __getitem__(self, idx):
    gold_feature, [], [], [], Q, A, Keywords_A, A_list, _, type1, _, qcate= self.instance_list[idx]
    # with open("/home/ubuntu/Project/11-777-Project-aa/Data/imgs.lineidx", "r") as fp_lineidx:
    #   lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
    # if type1 == 'img':
      
    #   f=[]
    #   f_count = len(gold_feature)
    #   for i in gold_feature:
    #   #   with open("/home/ubuntu/Project/11-777-Project-aa/Data/imgs.tsv", "r") as fp:
    #   #     fp.seek(lineidx[int(i)%10000000])
    #   #     imgid, img_base64 = fp.readline().strip().split('\t')
    #   #   f.append(img_base64)
    #   # while(len(f)<3):
    #   #   f.append(f[0])
    #   # print(f)
    #     with open(i, "rb") as f:
    #       features = pickle.load(f)
    #     img = features['fc1_features']

    #     f.append(img)
    #   while(len(f)<3):
    #     f.append(f[0])
    #   f = torch.stack(f)

    # else:
    #   # f=[]
    #   # for i in gold_feature:
    #   #   f.append(i)
    #   # while(len(f)<3):
    #   #   f.append(f[0])
    #   pass
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
    # tokenized_data1 = self.tokenizer(
    #     Q,
    #     return_tensors="pt",
    #     padding='max_length',
    #     truncation=True,
    #     max_length=32)
    # Q_tokens = tokenized_data1.input_ids[0]

    caption_len = tokenized_data.attention_mask[0].sum()

      
    return gold_feature, Q, tokens, caption_len,type1
  