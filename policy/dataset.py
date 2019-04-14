import torch.utils.data as data
import numpy as np

from torchvision.datasets.folder import default_loader
from PIL import Image

class ImageAction(data.Dataset):
    def __init__(self, root, test=False, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.data = []
        data = np.load(self.root)
        lim = len(data) // 5
        for i in range(len(data)): # Iterate through each trajectory
            if (test and i < lim) or (not test and i >= lim):
                for j in range(len(data[i]) - 1):
                    img_a = data[i][j][0]#.transpose(2, 0, 1)
                    img_b = data[i][j+1][0]#.transpose(2, 0, 1)
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                        imgs = (img_a, img_b)
                        action = data[i][j+1][1]['state'] - data[i][j][1]['state']
                        action = action[1:].reshape(6)
                        self.data.append((imgs, action))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        img_1 = self.data[index][0][0]
        img_2 = self.data[index][0][1]
        action = self.data[index][1]
        newimage = Image.new('RGB', (len(img_1), len(img_1)))
        newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        newimage = Image.new('RGB', (len(img_2), len(img_2)))
        newimage.putdata([tuple(p) for row in img_2 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return ((img_1, img_2), action)

    def __len__(self):
        return len(self.data)
