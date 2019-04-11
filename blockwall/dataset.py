import torch.utils.data as data
import numpy as np
import pickle as pkl
from PIL import Image
from torch.autograd import Variable
import torch
# from torchvision.datasets.folder import is_image_file, default_loader, find_classes, \
# IMG_EXTENSIONS
from torchvision.datasets.folder import default_loader, \
IMG_EXTENSIONS
import os
import os.path

from torchvision.utils import save_image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class MergedDataset(data.Dataset):
    """
    Merged multiple datasets into one. Sample together.
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        assert all(len(d) == self.__len__() for d in self.datasets)

    def __getitem__(self, index):
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])

    def __repr__(self):
        fmt_str = ''
        for dataset in self.datasets:
            fmt_str += dataset.__repr__() + '\n'
        return fmt_str

def make_dataset(dir, class_to_idx):
    actions = []
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            # if root[-2:] not in ['66', '67', '68']:
            for fname in sorted(fnames, key=(lambda x: int(x[:-4]))):
                if is_image_file(fname):
                    actions.append(0)
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                # if fname == 'actions.npy':
                #     path = os.path.join(root, fname)
                #     actions.append(np.load(path))
                #     actions[-1][-1, 4] = 0.0
            actions[-1] = 1
    return images, actions#np.concatenate(actions, axis=0)

def make_pair(imgs, resets, k, get_img, root):
    """
    Return a list of image pairs. The pair is picked if they are k steps apart,
    and there is no reset from the first to the k-1 frames.
    Cases:
        If k = -1, we just randomly pick two images.
        If k >= 0, we try to load img pairs that are k frames apart.
    """
    if k < 0:
        return list(zip(imgs, np.random.permutation(imgs)))

    filename = os.path.join(root, 'imgs_skippedd_%d.pkl' % k)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    image_pairs = []
    bad_pairs = []
    for i, img in enumerate(imgs):
        if np.sum(resets[i:i+k]) != 0:
            # import matplotlib.pyplot as plt
            # rg = np.linspace(-1, 1, 20)
            # fig = plt.figure()

            # a=fig.add_subplot(2, 1, 1)

            # x=plt.imshow(img, vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
            # #x=plt.imshow(image_recon, interpolation='none',cmap=plt.get_cmap('gray'))
            # a.xaxis.set_visible(False)
            # a.yaxis.set_visible(False)

            # a=fig.add_subplot(2, 1, 2)
            # x=plt.imshow(imgs[i+k], vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
            # #x=plt.imshow(image_recon, interpolation='none',cmap=plt.get_cmap('gray'))
            # a.xaxis.set_visible(False)
            # a.yaxis.set_visible(False)


            # plt.show()
            # plt.close()
            if i+k < len(imgs):
                bad_pairs.append((img, imgs[i+k]))
            else:
                print("OH NO")
        if np.sum(resets[i:i+k]) == 0 and (get_img(imgs[i+k][0])-get_img(img[0])).abs().max() > 0.5:
            image_pairs.append((img, imgs[i+k]))
            image_pairs.append((imgs[i+k], img))
    with open(filename, 'wb') as f:
        pkl.dump(image_pairs, f)
    with open('bad_pairs.pkl', 'wb') as f:
        pkl.dump(bad_pairs, f)
    return image_pairs

# class ImagePairs(data.Dataset):
#     """
#     A copy of ImageFolder from torchvision. Output image pairs that are k steps apart.
#
#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         n_frames_apart (int): The number of frames between the image pairs. Fixed for now.
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         img_pairs (list): List of pairs of (image path, class_index) tuples
#     """
#
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader, n_frames_apart=1):
#         classes, class_to_idx = find_classes(root)
#         imgs, actions = make_dataset(root, class_to_idx)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#         # resets = 1. - actions[:, -1]
#         resets = actions
#         print("len images then resets: {0}, {1}".format(len(imgs), len(resets)))
#         assert len(imgs) == len(resets)
#
#         self.root = root
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#         img_pairs = make_pair(imgs, resets, n_frames_apart, self._get_image, self.root)
#         self.img_pairs = img_pairs
#
#     def _get_image(self, path):
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         output = []
#         for path, target in self.img_pairs[index]:
#             img = self.loader(path)
#             if self.transform is not None:
#                 img = self.transform(img)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#             output.append((img, target))
#         return output
#
#     def __len__(self):
#         return len(self.img_pairs)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str

class ImagePairsOther(data.Dataset):
    """
    A copy of ImageFolder from torchvision. Output image pairs that are k steps apart.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, n_frames_apart=1):
        self.root = root
        self.data = []
        data = np.load(root)
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i]) - 1):
                img_a = data[i][j][0]
                img_b = data[i][j+1][0]
                if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                    imgs = (img_a, img_b)
                    self.data.append(imgs)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]
        newimage = Image.new('RGB', (len(img_1), len(img_1)))
        newimage.putdata([tuple(p) for row in img for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        newimage = Image.new('RGB', (len(img_2), len(img_2)))
        newimage.putdata([tuple(p) for row in img for p in row])
        if self.target_transform is not None:
            arr = np.array(self.target_transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return (img_1, img_2)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"

class ImageNumpyObst(data.Dataset):
    """
    Returns image pairs from a numpy file, also the obstacle image to be
    conditioned on, specifically stripe_rope.npy

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, test=False, transform=None, no_stripe=False, target_transform=None, loader=default_loader, gray=False, reverse=False):
        self.gray = gray
        if gray:
            self.root = root + '/stripe_rope_obst_bw.npy'
            if no_stripe:
                self.root = root + '/stripe_rope_obst_bwtip.npy'
        else:
            self.root = root + '/stripe_rope_obst.npy'
        self.data = []
        self.obsts = []
        data = np.load(self.root)
        for i in range(len(data)): # Iterate through each trajectory
            obst = data[i][0]

            # newimage = Image.new('RGB', (len(obst), len(obst)))
            # newimage.putdata([tuple(p) for row in obst for p in row])
            # if transform is not None:
            #     t_obst = np.array(transform(newimage))
            # else:
            t_obst = obst.transpose(2, 0, 1)
            diff = 255 / 2.
            t_obst = t_obst - diff
            t_obst = t_obst / diff
            # t_obst = t_obst / 255.

            self.obsts.append(t_obst)
            for j in range(1, len(data[i]) - 1):
                img_a = data[i][j]#.transpose(2, 0, 1)
                img_b = data[i][j+1]#.transpose(2, 0, 1)
                if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                    imgs = (img_a, img_b, obst)
                    #action = data[i][j+1][1]['action']
                    self.data.append(imgs)
                    if reverse:
                        rev = (img_b, img_a, obst)
                        self.data.append(rev)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # img = self.data[index]
        # # newimage = Image.new('RGB', (len(img), len(img)))
        # # newimage.putdata([tuple(p) for row in img for p in row])
        # # if self.transform is not None:
        # #     return np.array(self.transform(newimage))
        # return img
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]
        img_3 = self.data[index][2] 
        if self.gray:
            newimage = Image.new('L', (len(img_1), len(img_1)))
            newimage.putdata([p for row in img_1 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_1), len(img_1)))
            newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        if self.gray:
            newimage = Image.new('L', (len(img_2), len(img_2)))
            newimage.putdata([p for row in img_2 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_2), len(img_2)))
            newimage.putdata([tuple(p) for row in img_2 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        if self.gray:
            newimage = Image.new('L', (len(img_3), len(img_3)))
            newimage.putdata([p for row in img_3 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_3), len(img_3)))
            newimage.putdata([tuple(p) for row in img_3 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_3 = arr
        else:
            img_3 = img_3.transpose(2, 0, 1)
        return (img_1, img_2, img_3)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"

class ImageNumpy(data.Dataset):
    """
    Returns image pairs from a numpy file, specifically stripe_rope.npy

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, test=False, reverse=False, transform=None, no_stripe=False, target_transform=None, loader=default_loader, gray=False, k_steps=1):
        self.gray = gray
        self.data = []

        data = np.load(root)
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i]) - k_steps):
                if self.gray and j == 0:
                    continue
                if j % k_steps != 0:
                    continue
                img_a = data[i][j][0]#.transpose(2, 0, 1)
                img_b = data[i][j+k_steps][0]#.transpose(2, 0, 1)
                if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                    imgs = (img_a, img_b)
                    #action = data[i][j+1][1]['action']
                    self.data.append(imgs)
                    if reverse:
                        self.data.append((img_b, img_a))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # img = self.data[index]
        # # newimage = Image.new('RGB', (len(img), len(img)))
        # # newimage.putdata([tuple(p) for row in img for p in row])
        # # if self.transform is not None:
        # #     return np.array(self.transform(newimage))
        # return img
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]
        if self.gray:
            newimage = Image.new('L', (len(img_1), len(img_1)))
            newimage.putdata([p for row in img_1 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_1), len(img_1)))
            newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        if self.gray:
            newimage = Image.new('L', (len(img_2), len(img_2)))
            newimage.putdata([p for row in img_2 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_2), len(img_2)))
            newimage.putdata([tuple(p) for row in img_2 for p in row])
        # if self.target_transform is not None:
        #     arr = np.array(self.target_transform(newimage))
        #     img_2 = arr
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return (img_1, img_2)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        # fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        # fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        # fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # return fmt_str
        return "data"

class ImageAction(data.Dataset):
    """
    Returns image pairs along with the action between them

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, test=False, transform=None, target_transform=None, loader=default_loader):
        self.root = root + '/randact.npy'
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
                        #action = data[i][j+1][1]['action']
                        action = data[i][j+1][1]['state'] - data[i][j][1]['state']
                        action = action[1:].reshape(6)
                        self.data.append((imgs, action))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # img = self.data[index]
        # # newimage = Image.new('RGB', (len(img), len(img)))
        # # newimage.putdata([tuple(p) for row in img for p in row])
        # # if self.transform is not None:
        # #     return np.array(self.transform(newimage))
        # return img

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
        # if self.target_transform is not None:
        #     arr = np.array(self.target_transform(newimage))
        #     img_2 = arr
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return ((img_1, img_2), action)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        # fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        # fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        # fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # return fmt_str
        return "data"

class ImagePairsClassifier2(data.Dataset):
    """
    ImagePairs Classifier, but allows for smoothed labeling
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, smooth=False):
        self.root = root
        self.data = []
        data = np.load(root)
        count = 0
        x=0
        k=0
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i]) - 1):
                img_a = data[i][j][0]
                img_b = data[i][j+1][0]
                state_a = data[i][j][1]['state']
                state_b = data[i][j+1][1]['state']
                state_diff = np.absolute(np.sum(state_b - state_a, axis=1)) < .1
                if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                    count += 1
                    if smooth:
                        x = np.random.uniform(.8,1)
                    else:
                        x = 1
                    imgs = (img_a, img_b, x)
                    self.data.append(imgs)
        for i in range(len(data)):
            for j in range(len(data[i])):
                img_a = data[i][j][0]
                traj = np.random.randint(0, len(data))
                pos = np.random.randint(0, len(data[traj]))
                img_b = data[traj][pos][0]
                state_a = data[i][j][1]['state']
                state_b = data[traj][pos][1]['state']
                state_diff = np.absolute(np.sum(state_b - state_a, axis=1)) < .25
                state_diff_big = np.absolute(np.sum(state_b - state_a, axis=1)) < .65
                if np.sum(state_diff) >= 2 and np.sum(state_diff_big) == 3:
                    if smooth:
                        x = np.random.uniform(.8,1)
                    else:
                        x = 1
                    imgs = (img_a, img_b, x)
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0 and np.amax(img_a) > .2 and np.amax(img_b) > .2:
                        imgs = (img_a, img_b, x)
                        self.data.append(imgs)
                else:
                    if smooth:
                        x = np.random.uniform(0,.2)
                    else:
                        x = 0
                    imgs = (img_a, img_b, x)
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                        imgs = (img_a, img_b, x)
                        self.data.append(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        #print(len(self.data))

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]

        label = self.data[index][2]
        newimage = Image.new('RGB', (len(img_1), len(img_1)))
        newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        newimage = Image.new('RGB', (len(img_2), len(img_2)))
        newimage.putdata([tuple(p) for row in img_2 for p in row])
        if self.target_transform is not None:
            arr = np.array(self.target_transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return (img_1, img_2, label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"


class ImagePairsClassifier(data.Dataset):
    """
    ImagePairs Classifier, will output pairs right next to each other if asked to, otherwise random pairs as negative examples
    for blocks

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, closest=False, transform=None, target_transform=None, loader=default_loader,step=1, state_diff=.1):
        self.root = root
        self.data = []
        self.closest = closest
        data = np.load(root)
        count = 0
        p = 0
        n = 0
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i]) - step):
                img_a = data[i][j][0]
                img_b = data[i][j+step][0]
                state_a = data[i][j][1]['state']
                state_b = data[i][j+step][1]['state']
                state_diff = np.absolute(np.sum(state_b - state_a, axis=1)) < state_diff
                if closest:
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0 and np.amax(img_a) > .2 and np.amax(img_b) > .2:
                        count += 1
                        imgs = (img_a, img_b, 1.0)
                        self.data.append(imgs)
                else:
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                        count += 1
                        imgs = (img_a, img_b, 1.0)
                        p += 1
                        self.data.append(imgs)
        for i in range(len(data)):
            for j in range(len(data[i])):
                img_a = data[i][j][0]
                traj = np.random.randint(0, len(data))
                pos = np.random.randint(0, len(data[traj]))
                img_b = data[traj][pos][0]
                state_a = data[i][j][1]['state']
                state_b = data[traj][pos][1]['state']
                state_diff = np.absolute(np.sum(state_b - state_a, axis=1)) < .05
                state_diff_big = np.absolute(np.sum(state_b - state_a, axis=1)) < .3
                if np.sum(state_diff) >= 2 and np.sum(state_diff_big) == 3:
                    p+=1
                    lab = 1.0
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0 and np.amax(img_a) > .2 and np.amax(img_b) > .2:
                        imgs = (img_a, img_b, lab)
                        self.data.append(imgs)
                else:
                    lab = 0.0
                    n +=1
                    if np.sum(img_a) != 0 and np.sum(img_b) != 0:
                        imgs = (img_a, img_b, lab)
                        self.data.append(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        #print(len(self.data))
        print(p)
        print(n)
        print("+++++++++++++++++++++++++++++++++++")


        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]
        # print(img_1)
        if self.closest:
            diff = 255. / 2.
            img_1 = (img_1.transpose(1, 2, 0) * diff) + diff
            img_2 = (img_2.transpose(1, 2, 0) * diff) + diff
            img_1 = img_1.astype(int)
            img_2 = img_2.astype(int)
        label = self.data[index][2]
        newimage = Image.new('RGB', (len(img_1), len(img_1)))
        newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        newimage = Image.new('RGB', (len(img_2), len(img_2)))
        newimage.putdata([tuple(p) for row in img_2 for p in row])
        if self.target_transform is not None:
            arr = np.array(self.target_transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return (img_1, img_2, label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"

class ImagePairsClassifierRope(data.Dataset):
    """
    ImagePairs Classifier, will output pairs right next to each other if asked to, otherwise random pairs as negative examples
    for rope

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, gray=False, closest=False, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.data = []
        self.closest = closest
        data = np.load(root)
        count = 0
        start = 0
        self.gray = gray
        if gray:
            start = 1
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i]) - 1):
                if gray and j == 0:
                    continue
                img_a = data[i][j]
                img_b = data[i][j+1]
                count += 1
                imgs = (img_a, img_b, 1.0)
                self.data.append(imgs)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if gray and j == 0:
                    continue
                img_a = data[i][j]
                traj = np.random.randint(start, len(data))
                pos = np.random.randint(start, len(data[traj]))
                img_b = data[traj][pos]
                imgs = (img_a, img_b, 0.0)
                self.data.append(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        print(len(self.data))

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_1 = self.data[index][0]
        img_2 = self.data[index][1]
        if self.closest:
            diff = 255. / 2.
            img_1 = (img_1.transpose(1, 2, 0) * diff) + diff
            img_2 = (img_2.transpose(1, 2, 0) * diff) + diff
            img_1 = img_1.astype(int)
            img_2 = img_2.astype(int)
        label = self.data[index][2]
        if self.gray:
            newimage = Image.new('L', (len(img_1), len(img_1)))
            newimage.putdata([p for row in img_1 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_1), len(img_1)))
            newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        if self.gray:
            newimage = Image.new('L', (len(img_2), len(img_2)))
            newimage.putdata([p for row in img_2 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_2), len(img_2)))
            newimage.putdata([tuple(p) for row in img_2 for p in row])
        if self.target_transform is not None:
            arr = np.array(self.target_transform(newimage))
            img_2 = arr
        else:
            img_2 = img_2.transpose(2, 0, 1)
        return (img_1, img_2, label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"

class ImagePairsClassifierDomainRope(data.Dataset):
    """
    ImagePairs Classifier, will output pairs right next to each other if asked to, otherwise random pairs as negative examples
    for rope

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    def __init__(self, root, gray=False, generator=None, transition=None, closest=False, transform=None, target_transform=None, loader=default_loader):
        data_name = 'domain_rope.pkl'
        obst_name = 'obst_rope.pkl'
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.closest = closest
        self.gray = gray
        from pathlib import Path 
        if Path('./' + data_name).is_file() and Path('./' + obst_name).is_file():
            self.data = pkl.load(open(data_name, 'rb'))
            self.obsts = pkl.load(open(obst_name, 'rb'))
            return
        self.root = root
        self.data = []
        
        assert (generator is not None)
        data = np.load(root)
        print(data.shape)
        start = 0
        z_dim = generator.z_dim
        c_dim = generator.c_dim
        if gray:
            start = 1
        obsts = []
        
        for i in range(len(data)): # Iterate through each trajectory
            for j in range(len(data[i])):
                if gray and j == 0:
                    obsts.append(data[i][j])
                img_a = self.perturb(data[i][j])
                imgs = (img_a, .8)
                self.data.append(imgs)
        obsts = np.array(obsts)
        self.obsts = obsts
        batch = 64
        number = len(self.data) // 128 # double negative examples as positive if is // 64
        number = int(number * 1.5)
        pos = len(self.data)
        print("positive samples: {}".format(pos))   

        for j in range(number):
            random_z = Variable(torch.FloatTensor(batch, z_dim).normal_(0, 1).cuda())
            random_c = Variable(torch.FloatTensor(batch, c_dim).normal_(-1, 1).cuda())
            random_c_next = transition(random_c)
            obst_indices = np.random.randint(0, len(obsts), size=batch)

            torch_obst = Variable(torch.from_numpy(obsts[obst_indices].transpose(0, 3, 1, 2)).float().cuda())
            mid = 255 / 2.
            torch_obst = (torch_obst / mid) - 1.0
            fake, fake_next = generator(random_z, random_c, random_c_next, obst=torch_obst)
            fake = fake.data.cpu().numpy()
            fake_next = fake_next.data.cpu().numpy()

            for i in range(batch):
                img = np.clip(fake[i] + torch_obst[i].data.cpu().numpy() - (np.ones_like(fake_next[i])), -1, 1)
                img_next = np.clip(fake_next[i] + torch_obst[i].data.cpu().numpy() - (np.ones_like(fake_next[i])), -1, 1)
                img = img * mid + mid
                img_next = img_next * mid + mid
                img = img.transpose(1, 2, 0)
                img_next = img_next.transpose(1, 2, 0)
                self.data.append((img, 0.2))
                self.data.append((img_next, 0.2))
        print("negative samples: {}".format(len(self.data) - pos))
        pkl.dump(self.data, open(data_name, 'wb'))
        pkl.dump(self.obsts, open(obst_name, 'wb'))

        # self.data is going to be ((image1, image2), action)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_1 = self.data[index][0]
        if self.closest:
            diff = 255. / 2.
            img_1 = (img_1.transpose(1, 2, 0) * diff) + diff
            img_2 = (img_2.transpose(1, 2, 0) * diff) + diff
            img_1 = img_1.astype(int)
            img_2 = img_2.astype(int)
        label = self.data[index][1]
        if self.gray:
            newimage = Image.new('L', (len(img_1), len(img_1)))
            newimage.putdata([p for row in img_1 for p in row])
        else:
            newimage = Image.new('RGB', (len(img_1), len(img_1)))
            newimage.putdata([tuple(p) for row in img_1 for p in row])
        if self.transform is not None:
            arr = np.array(self.transform(newimage))
            img_1 = arr
        else:
            img_1 = img_1.transpose(2, 0, 1)
        return (img_1, label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "data"

    def perturb(self, img):
        import matplotlib.pyplot as plt
        # print("img max and min and shape: {0}, {1}, {2}".format(np.amax(img), np.amin(img), img.shape))
        # .25 times add noise throughout, .25 times don't change, .25 times add noise to rope, .25 times make rope completely random
        opt = np.random.randint(0, 4)
        sca = float(np.random.randint(1, 6))
        if opt == 0:
            noise = np.random.normal(scale=8.0, size=(64, 64, 1))
            rtn = np.clip(img+noise, 0, 255)
            return rtn
        elif opt == 1:
            for r in range(64):
                for c in range(64):
                    if img[r][c][0] < 250 and img[r][c][0] > 9:
                        img[r][c][0] += (np.random.normal(1, scale=4.0))
                    elif img[r][c][0] > 250:
                        img[r][c][0] += (np.random.normal(1, scale=sca))
            rtn = np.clip(img, 0, 255)
            rtn = img
            return img
        elif opt == 2:
            # print("opt 2")
            # rg = np.linspace(-1, 1, 20)
            # fig = plt.figure()
            # num_visualize = 25
            # batch_size = 100

            # a=fig.add_subplot(1, 2, 1)
            # label = opt
            # a.text(-1, -2, label, fontsize=8)
            # # print(np.amax(k), np.amin(k))
            # x=plt.imshow(img[:, :,0], vmin=0, vmax=255,  interpolation='none',cmap=plt.get_cmap('gray'))
            # #x=plt.imshow(image_recon, interpolation='none',cmap=plt.get_cmap('gray'))
            # a.xaxis.set_visible(False)
            # a.yaxis.set_visible(False)

            for r in range(64):
                for c in range(64):
                    if img[r][c][0] < 250 and img[r][c][0] > 9:
                        img[r][c][0] += (np.random.normal(1, scale=8.0))
                    elif img[r][c][0] > 250:
                        img[r][c][0] += (np.random.normal(1, scale=sca))
            rtn = np.clip(img, 0, 255)
            rtn = img

            # a=fig.add_subplot(1, 2, 2)
            # label = opt
            # a.text(-1, -2, label, fontsize=8)
            # # print(np.amax(k), np.amin(k))
            # x=plt.imshow(rtn[:, :,0], vmin=0, vmax=255,  interpolation='none',cmap=plt.get_cmap('gray'))
            # #x=plt.imshow(image_recon, interpolation='none',cmap=plt.get_cmap('gray'))
            # a.xaxis.set_visible(False)
            # a.yaxis.set_visible(False)

            # plt.show()
            # plt.savefig('./try.png')
            # plt.close()
            return rtn
        elif opt == 3:
            for r in range(64):
                for c in range(64):
                    if img[r][c][0] < 250 and img[r][c][0] > 10:
                        img[r][c][0] = np.random.randint(93, 163)
                    elif img[r][c][0] > 250:
                        img[r][c][0] += (np.random.normal(1, scale=sca))

            # rg = np.linspace(-1, 1, 20)
            # fig = plt.figure()
            # num_visualize = 25
            # batch_size = 100

            # a=fig.add_subplot(1, 2, 1)
            # label = opt
            # a.text(-1, -2, label, fontsize=8)
            # # print(np.amax(k), np.amin(k))
            # x=plt.imshow(img[:, :,0], vmin=0, vmax=255,  interpolation='none',cmap=plt.get_cmap('gray'))
            # #x=plt.imshow(image_recon, interpolation='none',cmap=plt.get_cmap('gray'))
            # a.xaxis.set_visible(False)
            # a.yaxis.set_visible(False)

            # plt.show()
            # plt.savefig('./try.png')
            # plt.close()

            return img
