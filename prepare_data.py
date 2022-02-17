import json
import os
from PIL import Image

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms

class SketchDataset(Dataset):
    def __init__(self, img_dir : str, annotations_file : str, device, preloaded_annotations=None, save_annotations=False):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.preloaded_annotations = preloaded_annotations
        self.save_annotations = save_annotations
        self.id_pairs = self.load_annotations()
        self.device = device

    def __len__(self):
        return len(self.id_pairs.keys())

    def __getitem__(self, index) -> tuple[th.Tensor, str]:
        id = list(self.id_pairs.keys())[index]
        image_path = os.path.join(self.img_dir, '{0:012d}.png'.format(int(id)))
        image = self.load_image(image_path)
        label = self.id_pairs[id]
        return image, label

    def load_annotations(self) -> dict:
        """
        Returns a dict of {image_id, caption} pairs for COCO dataset
        """
        if self.preloaded_annotations:
            with open(self.preloaded_annotations) as f:
                pairs = json.load(f)
        else:
            with open(self.annotations_file) as f:
                data = json.load(f)
            ann_dict_list = data['annotations']

            pairs = {}
            filenames = [f for f in os.listdir(self.img_dir) if os.path.isfile(f)]
            dir_size = len(filenames)
            for i, filename in enumerate(filenames):
                print(f'{i+1:d}/{dir_size:d}: Looking for {filename} in annotations', end="\r")
                for ann_dict in ann_dict_list:
                    if ann_dict['image_id'] == int(filename.strip("0").strip(".png")):
                        pairs[ann_dict['image_id']] = ann_dict['caption']
                        ann_dict_list.remove(ann_dict)
                        break

            if self.save_annotations:
                save_as_json = json.dumps(pairs)
                with open("pairs.json", "w") as f:
                    f.write(save_as_json)

        print("\n")
        print("Loaded annotations")
        return pairs

    def load_image(self, file_path : str) -> th.Tensor:
        """
        Returns the image loaded as a 1x256x256 image (for single channel only)
        """
        im = Image.open(file_path)
        img = transforms.ToTensor()(im)
        img = img.reshape(img.shape[1:])
        max_side = max(img.shape)
        pad_left, pad_top = max_side-img.shape[1], max_side-img.shape[0]
        padding = (pad_left//2, pad_top//2, pad_left//2+pad_left%2, pad_top//2+pad_top%2)
        img = transforms.Pad(padding, fill=1)(img)
        img = img.unsqueeze(0)
        img = transforms.Resize(256)(img)
        img = img.repeat(3, 1, 1).to(device=self.device)
        return img

# sketches = SketchDataset("./test_dir", "captions_val2017.json", 'cpu')
# for image, label in sketches:
#     print(image.shape)
