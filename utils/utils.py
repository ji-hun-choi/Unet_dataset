import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def info_print(txt):
    print("[INFO] ", txt)


def get_train_data(data_dir):
    img_path_list = []
    label_path_list = []

    input_path = os.path.join(data_dir, 'fire_img')
    label_path = os.path.join(data_dir, 'label2')
    # get label
    # label_df = pd.read_csv(data_dir+'/train.csv')
    label_path_list.extend(glob(os.path.join(label_path, '*.jpg')))
    label_path_list.sort()

    # get image path
    for label_path in tqdm(label_path_list):
        p = label_path.split('\\')[-1]
        input_p = os.path.join(input_path, p)
        img_path_list.append(input_p)

    return img_path_list, label_path_list


class CustomDataset(torch.utils.data.Dataset):
    # torch.utils.data.Dataset 이라는 파이토치 base class를 상속받아
    # 그 method인 __len__(), __getitem__()을 오버라이딩 해줘서
    # 사용자 정의 Dataset class를 선언한다
    def __init__(self, data_dir, img_path_list, label_path_list, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_input = img_path_list
        lst_label = label_path_list

        lst_input.sort()
        lst_label.sort()

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_label)

    # 여기가 데이터 load하는 파트
    def __getitem__(self, index):
        input_path = self.lst_input[index]
        label_path = self.lst_label[index]

        inputs = Image.open(input_path)
        label = Image.open(label_path)

        # normalize, 이미지는 0~255 값을 가지고 있어 이를 0~1사이로 scaling
        inputs = np.array(inputs) / 255.0
        label = np.array(label) / 255.0
        inputs = inputs.astype(np.float32)
        label = label.astype(np.float32)

        # 인풋 데이터 차원이 2이면, 채널 축을 추가해줘야한다.
        # 파이토치 인풋은 (batch, 채널, 행, 열)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]

        data = {'input': inputs, 'label': label}

        if self.transform:
            data['input'] = self.transform(data['input'])
            data['label'] = self.transform(data['label'])
        # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행
        return data


def split_data_set(path_lists, train_split_scale):
    img_path_list, label_path_list = path_lists[0], path_lists[1]
    info_print(f'img_path_list: {len(img_path_list)}'
               f'\n label_path_path:    {len(label_path_list)}')

    split_len = int(len(img_path_list) * train_split_scale)
    info_print(f'split_len: {split_len}')

    train_path_list = (img_path_list[:split_len], label_path_list[:split_len])
    valid_path_list = (img_path_list[split_len:], label_path_list[split_len:])

    return train_path_list, valid_path_list


def dataset(data_dir, path_list, transform, batch_size):
    dataset_train = CustomDataset(data_dir=data_dir, img_path_list=path_list[0],
                                  label_path_list=path_list[1], transform=transform)
    dataset_len = len(dataset_train)
    loader_dataset = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    return loader_dataset, dataset_len


def load_data_set(cfg):
    width, height, data_dir, img_scale, train_split_scale, batch_size = \
        cfg['width'], cfg['height'], cfg['data_dir'], cfg['img_scale'], cfg['train_split_scale'], cfg['batch_size']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([width, height]),
        transforms.Normalize(mean=img_scale, std=img_scale)
    ])

    info_print('All Data Load')
    path_list = get_train_data(data_dir)
    train_path_list, valid_path_list = split_data_set(path_list, train_split_scale)

    info_print('Train Data Load')
    loader_train_list = dataset(data_dir=data_dir, path_list=train_path_list,
                                transform=transform, batch_size=batch_size)

    info_print('Valid Data Load')
    loader_val_list = dataset(data_dir=data_dir, path_list=valid_path_list, transform=transform, batch_size=batch_size)

    return loader_train_list, loader_val_list


def save(ckpt_dir, model, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, f'{ckpt_dir}/model_epoch_{epoch}.pth')


