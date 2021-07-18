import json

with open("CMeIE_test_mid.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
flag = True
with open("CMeIE_test.json", 'w', encoding='utf-8') as f:
    for entry in data:
        if(flag):
            flag = False
        else:
            f.write('\n')
        json.dump(entry, f, ensure_ascii=False)
