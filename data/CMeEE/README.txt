1. Mainfest:
- CMeEE_train.json: 训练集 
- CMeEE_dev.json: 验证集
- CMeEE_test.json: 测试集, 选手提交的时候需要为每条记录增加"entities"字段，类型为列表。每个识别出来的实体必须包含"start_idx", "end_idx", "type"3个字段。如果该句子没有预测出实体，仍要增加"entities"字段，为空列表。
- example_gold.json: 标准答案示例
- example_pred.json: 提交结果示例
- README.txt: 说明文件

2. 评估指标以严格Micro-F1值为准

3. 该任务提交的文件名为：CMeEE_test.json
