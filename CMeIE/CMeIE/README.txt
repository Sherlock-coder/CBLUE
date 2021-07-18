1. Mainfest:
- 53_schema.json: SPO关系约束表
- CMeIE_train.json: 训练集 
- CMeIE_dev.json: 验证集
- CMeIE_test.json: 测试集,选手提交的时候需要为每条记录增加"spo_list"字段，类型为列表。每个识别出来的关系必须包含"subject", "predict", "object"3个字段，且"object"是一个字典（和训练数据保持一致）: {"@value": "some string"}。如果该句子没有预测出关系对，仍要增加"spo_list"字段，为空列表。
- example_gold.json: 标准答案示例
- example_pred.json: 提交结果示例
- README.txt: 说明文件

2. 评估指标以严格Micro-F1值为准

3. 该任务提交的文件名为：CMeIE_test.json
