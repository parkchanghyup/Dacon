import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정



def train_model(model, dataloaders_dict, optimizer, num_epochs, device, model_type):
    """
    train 데이터로 모델을 학습하고 valid 데이터로 모델을 검증하는 코드

    파라미터
    ---
    model :
        학습할 모델
    dataloaders_dict : dict
        train_dataloader과 validation_datalodaer가 들어 있는 dictonary
    optimizer :
        최적화 함수
    num_epochs : int
        학습 횟수
    device : cuda or cpu
        모델을 학습할 때 사용할 장비
    model_type : m or l or dense
        모델 back born

    returns
    best_model :
        검증데이터 셋 기준으로 가장 성능이 좋은 모델
    ---
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.9)
    model.to(device)
    torch.cuda.empty_cache()
    critrion = torch.nn.CrossEntropyLoss()
    # 학습이 어느정도 진행되면 gpu 가속화
    # torch.backends.cudnn.benchmark = False

    # loss가 제일 낮은 모델을 찾기위한 변수
    best_val_loss = int(1e9)
    for epoch in range(num_epochs):
        # epoch 별 학습 및 검증

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로
            else:
                model.eval()  # 모델을 추론 모드로

            epoch_loss = []  # epoch loss
            epoch_acc = []  # epoch 정확도
            for i, (imgs, labels) in enumerate(dataloaders_dict[phase]):
                images, labels = imgs, labels
                # tensor를 gpu에 올리기
                images = images.to(device)
                labels = labels.to(device)

                # 옵티마이저 초기화 초기화
                optimizer.zero_grad()

                # 순전파 계산
                with torch.set_grad_enabled(phase == 'train'):
                    probs = model(images)
                    loss = critrion(probs, labels)

                    # 학습시 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    probs = probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (labels == preds).mean()

                    epoch_loss.append(loss.item())
                    epoch_acc.append(batch_acc)

            # epoch별 loss 및 정확도
            epoch_loss = np.mean(epoch_loss)
            epoch_acc = np.mean(epoch_acc)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

            # 검증 오차가 가장 적은 최적의 모델을 저장
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model = model
                torch.save(best_model.state_dict(),
                           './checkpoint/' + '{}_{}_{}.pth'.format(model_type, epoch, best_val_loss))

            lr_scheduler.step()

    return best_model


dataframe = pd.read_csv('open/train.csv')
best_models = []
# 교차 검증을 진행할 K
K = 10
for model_type in ['m','l','dense']:
    for i in range(K):

        train_loader, val_loader = get_dataloader(
            dataframe, mode='TRAIN', batch_size=16)


        #  dict 형식으로 data loader 정의
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        num_epochs = 200
        max_grad_norm = 1
        learning_rate = 0.0005

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('사용하는 device :', device)
        print('--------------------', i+1, '/', K,
              '- fold start--------------------')

        model = BaseModel(model_type)

        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

        best_model = train_model(model, dataloaders_dict,
                                 optimizer, num_epochs, device, model_type)
        best_models.append(best_model)
