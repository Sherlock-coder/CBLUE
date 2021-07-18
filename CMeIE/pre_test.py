import json

path = "CMeIE/CMeIE_dev.json"

with open(path, 'r', encoding='utf-8') as f:
    sens = f.readlines()
data = []
for sen in sens:
    data.append(json.loads(sen.strip()))
with open("CMeIE_dev_raw.json", "w", encoding='utf-8') as f:
    json.dump(data, f,  ensure_ascii=False, indent=4, separators=(',', ': '))