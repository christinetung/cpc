import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

def binary_to_int(x):
    width = x.shape[1]
    return x.dot(2 ** np.arange(width)[::-1]).astype(np.int32)

def stochastic_binary_layer(x, tau=1.0):
    """
    x is (batch size)x(N) input of binary logits. Output is stochastic sigmoid, applied element-wise to the logits.
    """
    orig_shape = list(x.size())
    x0 = torch.zeros_like(x)
    x = torch.stack([x, x0], dim=2)
    x_flat = x.view(-1, 2)
    out = F.gumbel_softmax(x_flat, tau=tau, hard=True)[:, 0]
    return out.view(orig_shape)

def onehot_to_int(x):
    return np.where(x==1)[1]

def write_number_on_images(imgs, texts):
    """
    :param imgs: (numpy array) n x channel size x W x H
    :param texts: (numpy array) n
    :return: write texts on images.
    """
    n = texts.shape[0]
    for i in range(n):
        img = imgs[i]
        text = texts[i]
        trans_img = np.transpose(img, (1, 2, 0)).copy()
        # import ipdb;ipdb.set_trace()
        if type(text) == float or type(text) == np.float32:
            write_on_image(trans_img, "%.2f" % text)
        elif type(text) == str:
            write_on_image(trans_img, text)
        else:
            assert type(text) == int
            write_on_image(trans_img, "%d" % text)
        imgs[i] = np.transpose(trans_img, (2, 0, 1))

def write_on_image(img, text):
    """
    Make sure to write to final images - not fed into a generator.
    :param img: W x H x channel size
    :param text: string
    :return: write text on image.
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (2, 7)
    fontScale = 0.25
    fontColor = (1, 1, 1)
    lineType = 0
    cv2.putText(img,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
