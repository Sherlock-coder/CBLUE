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


re2sub = dict()
re2obj = dict()
with open("CMeIE/53_schemas.json", "r", encoding='utf-8') as f:
    sens = f.readlines()
schemas = []
for sen in sens:
    schemas.append(json.loads(sen.strip()))
for entry in schemas:
    re2sub[entry['predicate']] = entry['subject_type']
    re2obj[entry['predicate']] = entry['object_type']


def tag_extract(para, tags, texts, length):
    res = []
    for i in range(para['batch_size']):
        tag = tags[i]
        text = texts[i]['text']
        lst = []
        sub_lst = []
        j = 0
        while(j < len(text)):
            if (tag[j].startswith('B-S-')):
                type = tag[j].strip('B-S-')
                start = j
                j += 1
                while (j < length[i] and tag[j] == 'I'):
                    j += 1
                end = j
                if (end - start > 1):
                    sub_lst.append((type, text[start: end], start, end))
            else:
                j+=1
        j = 0
        while(j < len(text)):
            if(tag[j].startswith('B-O-')):
                type = tag[j].strip('B-O-')
                start = j
                j += 1
                while (j < length[i] and tag[j] == 'I'):
                    j += 1
                end = j
                sub_index_match = []
                sub_index = 0
                for i in range(len(sub_lst)):
                    if(sub_lst[i][0] == type):
                        sub_index_match.append(i)
                if(len(sub_index_match) > 0):
                    sub_index = min(sub_index_match, key = lambda x: abs(sub_lst[i][2] - start))
                if(len(sub_lst) > 0 and end - start > 1):
                    spo = dict()
                    sub_start = sub_lst[sub_index][2]
                    sub_end = sub_lst[sub_index][3]
                    subject = sub_lst[sub_index][1]
                    spo['Combined'] = ('。' in text[sub_end: start]) or ('。' in text[end: sub_start])
                    spo['predicate'] = type
                    spo['subject'] = subject
                    spo['subject_type'] = re2sub[type]
                    spo['object'] = {"@value": text[start: end]}
                    spo['object_type'] = {"@value": re2obj[type]}
                    lst.append(spo)
                # elif(len(sub_lst) == 0 and text.split('@')[0] != text):
                #     spo = dict()
                #     sub_start = 0
                #     sub_end = len(text.split('@')[0])
                #     subject = text[sub_start: sub_end]
                #     spo['Combined'] = ('。' in text[sub_end: start]) or ('。' in text[end: sub_start])
                #     spo['predicate'] = type
                #     spo['subject'] = subject
                #     spo['subject_type'] = re2sub[type]
                #     spo['object'] = {"@value": text[start: end]}
                #     spo['object_type'] = {"@value": re2obj[type]}
                #     lst.append(spo)
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
    return idx  #before sort -> after sort

if __name__ == "__main__":
    para = {'batch_size': 83, 'embedding_dim': 768, 'hidden_dim': 800, 'tagset_size': 91, 'dropout_rate': 0, 'epochs': 40}
    test_path = 'CMeIE_test_raw.json'
    myDataset = CustomDataset(test_path, mode='test')
    data = myDataset.data
    dataloader = DataLoader(myDataset, batch_size=para['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = BBC(para).cuda()
    model.load_state_dict(torch.load('params.pkl'))
    model.eval()
    f = codecs.open("CMeIE_test_mid.json", 'w', encoding='utf-8')
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
            res['spo_list'] = entities[i]
            res_out.append(res)
            # f.write(json.dumps(res, ensure_ascii=False, indent=2, separators=(',', ': ')))
        print("batch:%d" % index)
    f.write(json.dumps(res_out, ensure_ascii=False, indent=4, separators=(',', ': ')))


    # debug --> if you want to see the tagged sequence
    # g = codecs.open('res.txt', 'w', encoding='utf-8')
    # h = codecs.open('text.txt', 'w', encoding='utf-8')
    # for index, (samples, tags, length) in enumerate(iter(dataloader)):
    #     print("batch:%d" % index)
    #     prediction = model(samples=samples, length=length)
    #     texts = data[index * para['batch_size']:(index + 1) * para['batch_size']]
    #     idx = get_index(texts)
    #     texts_sorted = sorted(texts)
    #     pre_tag = [[idx2tag[i.item()] for i in id] for id in prediction]
    #     for i in range(para['batch_size']):
    #         real_idx = idx[i]
    #         h.write(texts[i]['text'] + '\n')
    #         g.write(str(pre_tag[real_idx]) + '\n')