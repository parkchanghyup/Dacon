

import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from datasets import CustomDataset



def get_split_data(df, config):
    """
    train 데이터를 8:2 비율로 train과 valid 데이터로 나누는 함수
    """
    train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=config['SEED'])

    return train_df, val_df


def get_dataloader(df, config, mode):
    """
    데이터프레임을 dataloader형태로 반환하는 함수
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # augmentations 적용
    train_augmentations = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        # transforms.RandomResizedCrop(224),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        # transforms.RandomGrayscale(p=0.4),
        # transforms.Grayscale(num_output_channels=3),
        # transforms.RandomAffine(45, shear=0.2),
        # transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        # transforms.Lambda(utils.randomColor),
        # transforms.Lambda(utils.randomBlur),
        # transforms.Lambda(utils.randomGaussian),
        transforms.ToTensor(),
        normalize,
    ])

    valid_augmentations = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        # transforms.CenterCrop(299),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])

    if mode == 'TRAIN':
        train, valid = get_split_data(df,config)

        # Data Loader
        train_dataset = CustomDataset(train, augmentations=train_augmentations)
        valid_dataset = CustomDataset(valid, augmentations=valid_augmentations)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=config['BATCH_SIZE'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=config['BATCH_SIZE'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return train_data_loader, valid_data_loader

    else:
        test_dataset = CustomDataset(df, augmentations=valid_augmentations)
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=config['BATCH_SIZE'],
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        return test_data_loader


def train_model(model, dataloaders_dict, optimizer, num_epochs, device):
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

        preds = []
        labels = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로
            else:
                model.eval()  # 모델을 추론 모드로

            epoch_loss = 0.0  # epoch loss
            epoch_corrects = 0  # epoch 정확도
            for i, batch in enumerate(dataloaders_dict[phase]):

                images, label = batch['image'], batch['label']
                # tensor를 gpu에 올리기
                images = images.to(device)
                label = label.to(device)

                # 옵티마이저 초기화 초기화
                optimizer.zero_grad()

                # 순전파 계산
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)
                    probs = torch.sigmoid(output)

                    loss = critrion(probs, label)

                    # 학습시 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 결과 계산
                    # loss계산
                    epoch_loss += loss.item() * len(probs)

                    # f1스코어 계산
                    probs = torch.argmax(probs, dim=1)

                    preds.extend(probs.cpu().detach().numpy())
                    labels.extend(label.cpu().detach().numpy())

            # epoch별 loss 및 f1-score

            epoch_loss = epoch_loss / len(dataloaders_dict[phase])
            epoch_f1 = f1_score(preds, labels, average="macro")

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} f1: {:.4f}'.format(epoch + 1, num_epochs,
                                                                          phase, epoch_loss, epoch_f1))

            # 검증 오차가 가장 적은 최적의 모델을 저장
            if not best_val_loss or epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model = model
                torch.save(best_model.state_dict(),
                           './checkpoint/' + '{}_{}_{}.pth'.format('mobilenetv3large', epoch, best_val_loss))

            lr_scheduler.step()

    return best_model
