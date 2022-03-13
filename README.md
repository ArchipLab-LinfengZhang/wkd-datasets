# Wavelet Knowledge Distillation for Efficient I2IT
This repository provides the dataset and codes in our cvpr2022 paper - wavelet knowledge distillation.

* * *

## Dataset
We have conducted quantitative experiments on Horse to Zebra, Edges to Shoes and Cityscapes, and qualitative experiments on Winter to Summer, Apple to Orange, Photo to Monet and Facades. 

* **Horse to Zebra** is an unpaired dataset which aims to transform images of horses to zebras and vice versa. It is built on the images from ImageNet. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip)
* **Edges to Shoes** is a paired dataset which aims to transform images of edges to natural images of shoes. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz)
* **Cityscapes** is firstly introduced as a segmentation & detection dataset. Following previous works, we use it in image-to-image translation tasks by reagrading the categories of different pixels as input, and the natural image as the output. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz)
* **Winter to Summer** is an unpaired dataset which aims to translate images from photos from winter to summer. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/summer2winter_yosemite.zip)
* **Apple to Orange** is an unpaired dataset which aims to translate images from apples to oranges. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip)
* **Photo to Monet** is an unpaired dataset which aims to translate natural photos to drawings of Monet. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip)
* **Facades** is a paired dataset which aims to translate images from the semantic segmentation of buildings to its correpsonding natural images. It can be downloaded from [here.](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz).

Please add the following citation if you use these datasets.
>@inproceedings{pix2pix,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}

for paired dataset and
>@inproceedings{cyclegan,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}

for unpaired dataset. You can find more details about these datasets from [here.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Codes
Our codes can be found in the supplementary material in CVPR website. Besides, we also provides the main implementation codes of wavelet transformation here.
Please download the *pytorch_wavelets* from [here.](https://github.com/fbcotter/pytorch_wavelets).

```
from pytorch_wavelets import DWTForward, DWTInverse
import torch

class WKD(wkd_level=4, wkd_basis='haar'):
    def __init__(self):
        self.xfm = DWTForward(J=wkd_level mode='zero',wave=wkd_basis)
    
    def get_wavelet_loss(self, student, teacher):
            student_l, student_h = self.xfm(student)
            teacher_l, teacher_h = self.xfm(teacher)
            loss = 0.0
            for index in range(len(student_h)):
                loss+= torch.nn.functional.l1_loss(teacher_h[index], student_h[index])
            return loss
```
