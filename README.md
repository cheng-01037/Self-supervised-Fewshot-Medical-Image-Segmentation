# [UNDER CONSTRUCTION] SSL_ALPNet

[ECCV'20] [Self-supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](https://arxiv.org/abs/2007.09886v1)

![](./intro.png)

**Abstract**:

Few-shot semantic segmentation (FSS) has great potential for medical imaging applications. Most of the existing FSS techniques require abundant annotated semantic classes for training. However, these methods may not be applicable for medical images due to the lack of annotations. To address this problem we make several contributions: (1) A novel self-supervised FSS framework for medical images in order to eliminate the requirement for annotations during training. Additionally, superpixel-based pseudo-labels are generated to provide supervision; (2) An adaptive local prototype pooling module plugged into prototypical networks, to solve the common challenging foreground-background imbalance problem in medical image segmentation; (3) We demonstrate the general applicability of the proposed approach for medical images using three different tasks: abdominal organ segmentation for CT and MRI, as well as cardiac segmentation for MRI. Our results show that, for medical image segmentation, the proposed method outperforms conventional FSS methods which require manual annotations for training.

**NOTE: This repository is still under construction**

If you find this code base useful, please cite our paper. Thanks!

```
@article{ouyang2020self,
  title={Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation},
  author={Ouyang, Cheng and Biffi, Carlo and Chen, Chen and Kart, Turkay and Qiu, Huaqi and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2007.09886},
  year={2020}
}
```

### 1. Setup

Install essential tools (see `requirements.txt`) 

```
jupyter==1.0.0
nibabel==2.5.1
numpy==1.15.1
opencv-python==4.1.1.26
Pillow==5.3.0
scikit-image==0.14.0
scipy==1.1.0
SimpleITK==1.2.3
tensorboardX==1.4
torch==1.3.0
torchvision==0.4.1
tqdm==4.32.2
```

### 2. Data Pre-processing and Pseudolabel Generation



### 3. Running



### 4. Acknowledgement

This code is based on vanilla [PANet] (https://github.com/kaixin96/PANet)(ICCV'19) by [Kaixin Wang](https://github.com/kaixin96. The data augmentation tools are from Dr. [Jo Schlemper](https://github.com/js3611)


