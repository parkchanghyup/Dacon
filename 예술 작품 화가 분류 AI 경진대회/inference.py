import pandas as pd
import numpy as np
import os

import torch
import torchvision.models as models

import random
from sklearn import preprocessing


def inference(best_models, test_dataloader):
    """
    test 데이터 셋을 inference 하는 코드

    파라미터
    ---
    best_models :
        k개의 fold에서 가장 성능 좋은 모델들

    test_dataloader : dataloader
        mini batch를 위한 data loader
    returns
    ---
    probs_list : list
        test 데이터의 예측값을 가지고 있는 list
    """
    
    probs_list = []


    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            # 추론
            model.eval()
            images = batch['image']
            images = images.to(device)
            output = model(images)
            probs = torch.sigmoid(output)
            probs = torch.argmax(probs, dim=1)
            probs = probs.cpu().detach().numpy()

        probs_list.extend(probs)

    return probs_list

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# test Dataset 정의
sample_submission = pd.read_csv('open/sample_submission.csv')
test_df = pd.read_csv('open/test.csv')
test_df['lables'] = 0
batch_size = 16
test_dataloader = get_dataloader(
    test_df, 'TEST', batch_size=batch_size)

# 저장된 모델 불러오기
model = models.efficientnet_b0(pretrained = False)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=50)
model.load_state_dict(torch.load('checkpoint/mobilenetv3large_93_24.4446646340204.pth'))
model = model.to(device)

prob_list = inference(model,test_dataloader)

train_df = pd.read_csv('open/train.csv')

# Label Encoding
le = preprocessing.LabelEncoder()
train_df['artist'] = le.fit_transform(train_df['artist'].values)

# sample_sumbissions 생성
sample_submission['artist'] = le.inverse_transform(prob_list).tolist()
sample_submission.to_csv('submission.csv',index = False)
