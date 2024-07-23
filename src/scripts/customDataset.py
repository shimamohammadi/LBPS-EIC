import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, x_path, ref_dir, dis_dir, transform=None):

        df_x = pd.DataFrame(x_path)
        self.dis_dir = dis_dir
        self.ref_dir = ref_dir
        self.ref_names = df_x['ref image']
        self.dist_1_names = df_x['distorted image A']
        self.dist_2_names = df_x['distorted image B']
        self.transform = transform

    # Loading a single datapoint
    def __getitem__(self, index):
        ref = Image.open(os.path.join(self.ref_dir,
                                      self.ref_names.iloc[index]))
        dis_1 = Image.open(os.path.join(self.dis_dir,
                                        self.dist_1_names.iloc[index]))
        dis_2 = Image.open(os.path.join(self.dis_dir,
                                        self.dist_2_names.iloc[index]))

        if self.transform is not None:
            ref = self.transform(ref)
            dis_1 = self.transform(dis_1)
            dis_2 = self.transform(dis_2)

        return ref, dis_1, dis_2

    def __len__(self):
        return self.ref_names.shape[0]


custom_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
