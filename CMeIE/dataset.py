import json

import torch as t
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('./')
from preprocess import getdata, get_test_data
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy


class CustomDataset(Dataset):
    def __init__(self, path, mode):
        if(mode == 'test'):
            self.data = get_test_data(path)
        elif(mode == 'train'):
            self.data = getdata(path)
        with open('tag2idx.json', 'r', encoding='utf-8') as f:
            self.tag2idx = json.load(f)
        self.idx2tag = dict()
        for key in self.tag2idx:
            self.idx2tag[self.tag2idx[key]] = key
        model_name = 'bert-base-multilingual-cased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        text, tag = self.data[item]['text'], self.data[item]['tag']
        # text = self.tokenizer.tokenize(text)
        text = text if len(text) <= 512 else text[: 512]
        tag = tag if len(tag) <=512 else tag[: 512]
        text = list(text)
        a = self.tokenizer.convert_tokens_to_ids(text)
        a = t.tensor([a])
        with t.no_grad():
            a = self.bert_model(a)
        a = a[0].squeeze(0).cuda() #第一层是什么?句嵌入吗?
        tag = [self.tag2idx[item] for item in tag]
        return a, tag

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    length = [len(x[1]) for x in data]
    samples = [x[0] for x in data]
    tags = [t.tensor(x[1]).cuda() for x in data]
    samples = pad_sequence(samples, batch_first=True, padding_value=0)
    tags = pad_sequence(tags, batch_first=True, padding_value=0)
    return samples, tags, length

if __name__ == "__main__":
    path = 'CMeEE/CMeEE/CMeEE_test.json'
    myDataset = CustomDataset(path)
    dataloader = DataLoader(myDataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    samples, tags, length= next(iter(dataloader))
    samples_pack = pack_padded_sequence(samples, length, batch_first=True)
    tags_pack = pack_padded_sequence(tags, length, batch_first=True)
