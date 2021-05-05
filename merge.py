import joblib
import numpy as np
import pandas as pd


roberte_wwm_ext_large = np.array(joblib.load('chinese-roberta-wwm-ext-large.pkl'))
roberte_wwm_ext = np.array(joblib.load('chinese-roberta-wwm-ext.pkl'))
bert_wwm_ext = np.array(joblib.load('chinese-bert-wwm-ext.pkl'))
#electra_180g_base_discriminator = np.array(joblib.load('chinese-electra-180g-base-discriminator.pkl'))

#predictions = (roberte_wwm_ext_large + roberte_wwm_ext + bert_wwm_ext + electra_180g_base_discriminator) / 4.
predictions = roberte_wwm_ext_large * 0.33984479529033995 + roberte_wwm_ext * 0.32860583355632844 + bert_wwm_ext * 0.33154937115333166
predictions = np.mean(predictions,0).argmax(1)
print(predictions.shape)

sub = pd.read_csv('data/sample.csv',dtype=object) #提交
sub['label'] = predictions
sub['label'] = sub['label'].apply(lambda x:['A','B','C','D'][x])

sub.to_csv('merged.csv',index=False)

