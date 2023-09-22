from data_processing import pre_proceccing, get_data
from trainer import train_inference, train_validation
import pandas as pd

# 데이터 로딩 및 전처리
train, test = get_data()
train_df = pre_proceccing(train)
test_df = pre_proceccing(test)

smape = train_validation(train_df)
print(f'train 데이터 SMAPE : {smape}')

preds = train_inference(train_df, test_df)

submission = pd.read_csv('./data/sample_submission.csv')
submission['answer'] = preds
submission.to_csv('submission.csv',index= False)
