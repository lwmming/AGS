# AGS
Code for the AAAI 2024 paper: "AGS: Affordable and Generalizable Substitute Training for Transferable Adversarial Attack" (accepted).

## Prerequisites
The repo requires the following:

1. **PyTorch** (version 1.9.0 or higher): The repo was tested on PyTorch 1.9.0 with CUDA 11.1 support.
2. **Hardware**: We have performed experiments on NVIDIA GeForce RTX 3090Ti with 24GB GPU memory. Similar or higher specifications are recommended for optimal performance.
3. **Python packages**: Additional Python packages specified in the `requirements.txt` file are necessary. Instructions for installing these are given below.

## Setup Instructions
```
conda create --name env_name python=3.7.6
source activate env_name
pip install -r requirements.txt
```

## Data Preparation
**Training Data:**

We train our AGS on three unlabeled & out-of-domain datasets:
  * [Paintings](https://www.kaggle.com/c/painter-by-numbers)
  * [Comics](https://www.kaggle.com/cenkbircanoglu/comic-books-classification)
  * [CoCo-2017(41k)](https://cocodataset.org/#download)

To enhance data input speed, we convert the data into LMDB format before training. You can access our LMDB format data via the following links:
  * [Comics-lmdb](https://drive.google.com/drive/folders/1juhde7RPtKDkNn64_r3fkn19wUi5WAgL?usp=drive_link)
  * [CoCo-2017(41k)-lmdb](https://drive.google.com/drive/folders/1ct6UeFJo50z-x2UvRoC_kQee0wxkY530?usp=drive_link)

After downloading, please place them into the './data' dir (The LMDB data of Paintings is too large, about 36G. It exceeds the size limit of my Google Cloud disk, so the upload cannot be successful. To this end, we give the process code, i.e., `folder2lmdb.py`. Using it, you can convert image files to LMDB format by yourself). 
You can alternatively utilize either the `ImageFolderTriple` class or the `ElementwiseTriple` class, both of which are provided in the `utils.py` file, to read training data from image files.

**Evaluation Data:**

The evaluation data is consistent with related works [Practical No-box Adversarial Attacks against DNNs](https://github.com/qizhangli/nobox-attacks) and [Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations](https://github.com/HashmatShadab/APR). Concretely, 5000 images are selected from ImageNet-Val (10 each from the first 500 classes). The `./data/selected_data.csv` is used by the `our_dataset.py` to load the selected 5000 images from the dataset.

## AGS Training
For instance, when training AGS on the COCO dataset:
```
python train_ags_lmdb.py --mode ags --eps_train 1.0 --inner_step 1 --wd 1e-4 --lr 0.1 --end_epoch 100 --data_dir your path to CoCo41K-lmdb/CoCo41K.lmdb --save_dir ./checkpoints/coco_ags --gpu 6
```
You can set the `--data_dir` accordingly to train on coco, comics and paintings.

We did not fully adjust the training hyperparameters of the model. If larger batch_size, longer iteration rounds, or more advanced optimization strategies are used, the performance of AGS still has great room for improvement. Welcome to try!

## Cross-Paradigm Attack (Evaluation)
Once the substitute model is trained, based on which, adversarial examples are crafted for the selected 5000 ImageNet-Val images. And then these adversarial examples are directly fed to various target models to get top-1 accuracy. The lower, the better.
```
python attack_evaluation.py --n_imgs 20 --batch_size 80 --model_dir ./checkpoints/coco_ags/models --model_pth ags_100.pth --gpu 6
```
You can set different substitute models, e.g., pretrained ResNet50 from torchvision, in attack_evaluation.py to obtain corresponding transfer-rate.
Moreover, you can also set more advanced target models to test the adversarial transferability of our AGS model.

## Our Pretrained Substitute Models
Here, we release the checkpoint of our trained AGS models.
| Dataset   |                                               AGS                                               |
|:----------|:----------------------------------------------------------------------------------------------------:|
| CoCo      |   [coco_ags_100.pth](https://drive.google.com/file/d/1k0MtvYZDGfFvpOz_HF-XIyEKQoR8ivAk/view?usp=drive_link)    | 
| Paintings | [paintings_ags_100.pth](https://drive.google.com/file/d/1hIkn4T9vnaVRKXCXTsfBeaQApKY45EOZ/view?usp=drive_link) |
| Comics    |  [comics_ags_100.pth](https://drive.google.com/file/d/1NB8zuYWxjHmLsAX2mt9MXsU5P-qUcuTU/view?usp=drive_link)   |

## Contact
Feel free to contact me! Any discussions and suggestions are welcome! rkwang@buaa.edu.cn

## References
Our code is patially based on [ Practical No-box Adversarial Attacks against DNNs](https://github.com/qizhangli/nobox-attacks) and [Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations](https://github.com/HashmatShadab/APR) repository. We really appreciate for the released code.






