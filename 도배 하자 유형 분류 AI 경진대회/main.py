import torch
import random
import os
import numpy as np
from data_processing import CreateDataloader
from model import customModel
from train import train_model
import glob
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


train_data_paht = './data/train/*/*'
num_class = len(glob.glob('./data/train/'))
CreateDataloader = CreateDataloader(train_data_paht, CFG)

best_models = []
# 교차 검증을 진행할 K
K = 1

for i in range(K):

    train_loader, val_loader = CreateDataloader.get_dataloader(
        mode='TRAIN')


    #  dict 형식으로 data loader 정의
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print('사용하는 device :', device)
    print('--------------------', i+1, '/', K,
          '- fold start--------------------')

    model = customModel(num_class)


    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG['LEARNING_RATE'])

    best_model = train_model(model, dataloaders_dict,
                             optimizer, CFG['EPOCHS'], device)
    best_models.append(best_model)




