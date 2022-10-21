import pandas as pd
import numpy as np
import os

import torch
import torchvision.models as models

import random
from sklearn import preprocessing
from utils import *

config = {
    'IMG_SIZE':256,
    'EPOCHS':100,
    'LEARNING_RATE':0.0005,
    'BATCH_SIZE':64,
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config['SEED']) # Seed 고정
print('----------seed 고정 완료----------')


best_models = []

df = pd.read_csv('./open/train.csv')

# Label Encoding
le = preprocessing.LabelEncoder()
df['artist'] = le.fit_transform(df['artist'].values)
print('----------label encode 완료----------')
################
# 교차 검증을 진행할 K
K = 1
for i in range(K):
    train_loader, val_loader = get_dataloader(
        df, config, mode='TRAIN'
    )
    #  dict 형식으로 data loader 정의
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    num_epochs = config['EPOCHS']
    max_grad_norm = 1
    learning_rate = config['LEARNING_RATE']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('사용하는 device :', device)
    print('--------------------', i + 1, '/', K,
          '- fold start--------------------')

    model = models.efficientnet_b4(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_model = train_model(model, dataloaders_dict,
                             optimizer, num_epochs, device)
    best_models.append(best_model)