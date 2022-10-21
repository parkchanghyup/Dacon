import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 meta_df,
                 augmentations=None):
        self.meta_df = meta_df  # 데이터의 인덱스와 정답지가 들어있는 DataFrame
        self.augmentations = augmentations  # Augmentation

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        path = str(self.meta_df.iloc[index, 1])
        image = Image.open('./open/' + path).convert('RGB')

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = int(self.meta_df.iloc[index, 2])
        sample = {'image': image, 'label': label}

        # augmentation 적용
        if self.augmentations:
            sample['image'] = self.augmentations(sample['image'])

        # sample 반환
        return sample