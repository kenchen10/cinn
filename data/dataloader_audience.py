import cv2, torch, torch.nn as nn
import numpy as np
import kornia as k, os, glob
from data.augmentation import Augmentation_random_crop
from glob import glob

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):

        self.data_path = opt.Data['data_path']

        self.seq_length = opt.Data['sequence_length']
        self.do_aug = opt.Data['aug']

        video_list = []
        self.videos = glob(self.data_path + '/*.pt')
        for vid in self.videos:
            if len(torch.load(vid)) > 16:
                video_list += [vid]
        self.videos = video_list
        if mode == 'train':
            self.videos = self.videos[0:int(.8*len(self.videos))]
        else:
            self.videos = self.videos[int(.8*len(self.videos)):]
        self.length = len(self.videos)

        self.videos = video_list
        self.length = len(self.videos)
        # print(self.length)

        if mode == 'train' and self.do_aug:
            self.aug = Augmentation_random_crop(opt.Data['img_size'], opt.Data.Augmentation)
        else:
            self.aug = torch.nn.Sequential(
                        k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                        k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video  = self.videos[idx]
        frames = torch.load(video)

        ## Sample random starting point in the sequence
        start_rand = np.random.randint(0, len(frames) - self.seq_length + 1)

        seq = torch.stack([frames[start_rand + i] for i in range(self.seq_length)], dim=0)
        # print(self.aug(seq).size())
        return {'seq': self.aug(torch.squeeze(seq))}
