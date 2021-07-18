import torch
import torch as t
from torchcrf import CRF
from torch import nn
import sys
sys.path.append('./')
from preprocess import getdata,get_test_data
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset, collate_fn
from bbc import BBC
from torch import optim
import codecs
import json
def tag_extract(para, tags, texts, length):
    res = []
    for i in range(para['batch_size']):
        tag = tags[i]
        text = texts[i]['text']
        j = 0
        lst = []
        while(j < length[i]):
            if(tag[j].endswith('-B')):
                tmp = dict()
                type = tag[j].rstrip('-B')
                tmp['start_idx'] = j
                j += 1
                while(j < length[i] and tag[j] == type + '-I'):
                    j += 1
                tmp['end_idx'] = j-1
                tmp['type'] = type
                tmp['entity'] = text[tmp['start_idx']:j]
                if(tmp['end_idx'] > tmp['start_idx']):
                    lst.append(tmp)
            else:
                j += 1
        res.append(lst)
    return res

def sorted(data):
    data = list(data)
    data.sort(key = lambda x: len(x['text']), reverse=True)
    return data

def get_index(data):
    data = list(zip(data, range(len(data))))
    data.sort(key=lambda x: len(x[0]['text']), reverse=True)
    tmp = list(zip(*data))[1]
    idx = dict()
    for i in range(len(tmp)):
        idx[tmp[i]] = i
    return idx

if __name__ == "__main__":
    para = {'batch_size': 40, 'embedding_dim': 768, 'hidden_dim': 400, 'tagset_size': 20, 'dropout_rate': 0, 'epochs': 40}
    test_path = 'CMeEE/CMeEE/CMeEE_test.json'
    myDataset = CustomDataset(test_path, mode='test')
    data = myDataset.data
    dataloader = DataLoader(myDataset, batch_size=para['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = BBC(para)
    model.load_state_dict(torch.load('params.pkl'))
    model.eval()
    f = codecs.open("CMeEE_test.json", 'w', encoding='utf-8')
    # g = codecs.open('res.txt', 'w', encoding='utf-8')
    # h = codecs.open('text.txt', 'w', encoding='utf-8')
    tag2idx, idx2tag = myDataset.tag2idx, myDataset.idx2tag
    res_out = []
    for index, (samples, tags, length) in enumerate(iter(dataloader)):
        prediction = model(samples=samples, length=length)
        prediction = [[idx2tag[x.item()] for x in i]for i in prediction]
        texts = data[index*para['batch_size']:(index+1)*para['batch_size']]
        idx = get_index(texts)
        texts_sorted = sorted(texts)
        after_sort = tag_extract(para, prediction, texts_sorted, length)
        entities = [after_sort[idx[i]] for i in range(para['batch_size'])]
        for i in range(para['batch_size']):
            res = dict()
            res['text'] = texts[i]['text']
            res['entities'] = entities[i]
            res_out.append(res)
            # f.write(json.dumps(res, ensure_ascii=False, indent=2, separators=(',', ': ')))
        print("batch:%d" % index)
    f.write(json.dumps(res_out, ensure_ascii=False, indent=4, separators=(',', ': ')))
    # for index, (samples, tags, length) in enumerate(iter(dataloader)):
    #     idx = get_index(data[index*para['batch_size']:(index+1)*para['batch_size']])
    #     print("batch:%d" % index)
    #     prediction = model(samples=samples, length=length)
    #     for i in range(para['batch_size']):
    #         h.write(data[index*para['batch_size']+idx[i]]['text']+'\n')
    #         for j in range(length[i]):
    #             g.write(str(prediction[i][j].item())+' ')
    #         g.write('\n')
