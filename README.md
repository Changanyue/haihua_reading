# haihua_reading
海华阅读理解比赛26名

1. train
python main.py
model分别设置为hfl/chinese-roberta-wwm-ext-large、hfl/chinese-roberta-wwm-ext、hfl/chinese-bert-wwm-ext，max_len都是256，batch_size根据GPU显存设置

2. test
python main.py

3. merge
python merge.py

备注：
1. 尝试过electra模型，效果较差
2. 尝试过batch_size为512，一方面难以收敛，另一方面结果不如256的
3. 尝试过随意交换选型，结果会变差
4. 尝试过限制文章内容、问题、答案各自的长度，结果会变差
