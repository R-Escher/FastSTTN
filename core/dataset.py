import json
import os
import random
import torch
import torchvision.transforms as transforms

from core.utils import (GroupRandomHorizontalFlip, Stack, ToTorchFormatTensor,
                        ZipReader, create_random_shape_with_random_motion)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug or split != 'train':
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        all_masks = create_random_shape_with_random_motion(len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        # load the 5 randomly chosen frames
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length):
    """ Chooses randomly 5 sequential frames half of the time (e.g. 11, 12, 13, 14, 15), 
        and the other half chooses 5 non-sequential random frames (e.g. 9, 50, 71, 80, 99). 
    """
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
