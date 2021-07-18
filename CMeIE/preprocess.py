import json

def getdata(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = []
    for entry in data:
        tmp = dict()
        for key in entry:
            if(key == 'text'):
                tmp[key] = entry[key]
            elif(key == 'entities'):
                tmp['tag'] = ['O' for i in range(len(entry['text']))]
                lst = entry[key]
                for item in lst:
                    tmp['tag'][item['start_idx']] = 'B-' + item['type']
                    for j in range(item['start_idx']+1, item['end_idx']+1):
                        tmp['tag'][j] = 'I'
        res.append(tmp)
    return res

# with open('res.json', 'w', encoding='utf-8') as f:
#     json.dump(res, f)

def get_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = []
    for entry in data:
        tmp = dict()
        tmp['text'] = entry['text']
        tmp['tag'] = ['O' for _ in range(len(entry['text']))]
        res.append(tmp)
    return res

if __name__ == "__main__":
    path = 'CMeEE/CMeEE/CMeEE_train.json'
    a = getdata(path)
    for i in range(10):
        print(a[i]['text'], a[i]['tag'])
        print(len(a[i]['text']), len(a[i]['tag']))
