import numpy as np
import torch


import warnings
warnings.filterwarnings(action='ignore')


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
                           './checkpoint/' + '{}_{}_{}.pth'.format(epoch, best_val_loss))

            lr_scheduler.step()

    return best_model

