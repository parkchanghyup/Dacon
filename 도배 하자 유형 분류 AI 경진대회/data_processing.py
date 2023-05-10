import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from UniformAugment import UniformAugment
from sklearn import preprocessing
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        # img_path = './data'+ self.img_path_list[index][1:]
        img_path = self.img_path_list[index]
        image = Image.open(img_path)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)


class CreateDataloader:
    def __init__(self, path, CFG):

        self.all_img_list = glob.glob(path)

        self.df = pd.DataFrame(columns=['img_path', 'label'])
        self.df['img_path'] = self.all_img_list
        self.df['label'] = self.df['img_path'].apply(lambda x : str(x).split('/')[2])

        self.CFG = CFG

    def get_dataloader(self, mode):
        """
        데이터프레임을 dataloader형태로 반환하는 함수
        """

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # augmentations 적용
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #RandAugment(),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform.transforms.insert(0, UniformAugment())

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        if mode == 'TRAIN':

            train, val, _, _ = train_test_split(self.df, self.df['label'], test_size=0.2, stratify=self.df['label'],
                                                random_state=self.CFG['SEED'])

            le = preprocessing.LabelEncoder()
            train['label'] = le.fit_transform(train['label'])
            val['label'] = le.transform(val['label'])

            train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
            val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)

            train_loader = DataLoader(train_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

            return train_loader, val_loader

        else:
            test_dataset = CustomDataset(self.df['img_path'].values, None, test_transform)
            test_loader = DataLoader(test_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
            return test_loader
