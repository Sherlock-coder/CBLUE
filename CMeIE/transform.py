import re
import json

path = "CMeIE/CMeIE_train.json"

with open(path, 'r', encoding='utf-8') as f:
    sens = f.readlines()
data = []
for sen in sens:
    data.append(json.loads(sen.strip()))
res = []
for entry in data:
    tmp = dict()
    tmp['text'] = entry['text']
    tmp['entities'] = []
    for spo in entry['spo_list']:
        try:
            entity = dict()
            begin, end = re.search(spo['subject'], tmp['text']).span()
            entity['start_idx'] = begin
            entity['end_idx'] = end - 1
            entity['type'] = 'S-' + spo['predicate']
            entity['entity'] = tmp['text'][begin: end]
            tmp['entities'].append(entity)
            entity = dict()
            begin, end = re.search(spo['object']['@value'], tmp['text']).span()
            entity['start_idx'] = begin
            entity['end_idx'] = end - 1
            entity['type'] = 'O-' + spo['predicate']
            entity['entity'] = tmp['text'][begin: end]
            tmp['entities'].append(entity)
        except:
            print("exception")
    res.append(tmp)
with open("CMeIE/CMeEE_train.json", 'w', encoding='utf-8') as f:
    json.dump(res, f,  ensure_ascii=False, indent=4, separators=(',', ': '))
lst = ["MASK", "O", "I"]
for entry in res:
    for entity in entry['entities']:
        if('B-' +entity['type'] not in lst):
            lst.append('B-' + entity['type'])
idx2tag = dict()
tag2idx = dict()
for i in range(len(lst)):
    idx2tag[i] = lst[i]
    tag2idx[lst[i]] = i
with open("idx2tag.json", "w", encoding='utf-8') as f:
    json.dump(idx2tag, f, ensure_ascii=False, indent=4, separators=(',', ': '))
with open("tag2idx.json", "w", encoding='utf-8') as f:
    json.dump(tag2idx, f, ensure_ascii=False, indent=4, separators=(',', ': '))