import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from read_yaml import parse_yaml

# 1.create dataset
class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),# convert PIL.Image to tensor, which is GY
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalization
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx:int):
        # img to tensor, label to tensor
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)

        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog':
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # str-->int
        label = torch.as_tensor(label, dtype=torch.int64) # must use long type, otherwise raise error when training, "expect long"
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)

# 2.dataset split
def dataset_split(full_ds, train_rate):
    train_size = int(len(full_ds) * train_rate)
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds

# 3. data loader
def dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return data_loader


# this function is only for debugging
def debug():
    yaml_path = './config.yaml'
    cfg = parse_yaml(yaml_path)
    # print(cfg)
    train_path = cfg['train_path']
    test_path = cfg['test_path']
    batch_size = cfg['batch_size']
    # print(train_path)
    train_ds = MyDataset(train_path)
    new_train_ds, validation_ds = dataset_split(train_ds, 0.8)
    test_ds = MyDataset(test_path, train=False)
    new_train_loader = dataloader(new_train_ds, batch_size)
    validation_loader = dataloader(validation_ds, batch_size)
    test_loader = dataloader(test_ds, batch_size)

    print(len(new_train_ds))
    print(len(test_ds))

    # testing train data can iterate or not
    for i, item in enumerate(tqdm(new_train_ds)):
        print(item)
        break

    # testing data shape
    for i, item in enumerate(new_train_loader):
        print(item[0].shape) # torch.Size([32, 3, 224, 224]), data dim from 3 to 4 (batch_size, C, H, W),
        # and the more dim is batch_size
        print(item[0])
        print(item[1]) # tensor([1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])

        print(item[0].size(0)) # 32
        print(item[0].size(1)) # 3
        print(item[0].size(2)) # 224
        print(item[0].size(3)) # 224
        break




if __name__ == '__main__':
    debug()