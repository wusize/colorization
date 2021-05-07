# Colorization
This is the code for the colorization project of the National Innovation Program.

## Introduction
We use CNN models to colorize gray-sacle images in an end-to-end manner. Our work is primarily inspired by [Let there be color](https://github.com/satoshiiizuka/siggraph2016_colorization) which uses two parallel CNN sub-modules to extract low-level and high-level features. As a follow-up, we change the CNN-based backbone to the prevalent ResNet50 and use the [pre-trained parameters](https://download.pytorch.org/models/resnet50-0676ba61.pth) to facilitate the training process.

Besides, we add a CNN-based attention module and appendix a discriminator to the colorizer in an attempt to get more plausible results.

## Requirements

```
pytorch
torchvision
scikit-image
numpy
tqdm
```
## Data
```
${DATA_DIR}
|-- 0000
    |-- 0000_00001.jpg
    |-- 0000_00002.jpg
    ......
|-- 0001
|-- 0002
......
```

## Pre-trained model
To get the pre-trained resnet50 model, download the .pth file from
```
https://download.pytorch.org/models/resnet50-0676ba61.pth
```
Note that if you are using an old version of Pytorch (e.g. 1.3) on your remote server, remember to transform the model on your PC as follows.
```
import torch
m = torch.load(${MODEL_DIR})
torch.save(m, ${MODEL_DIR}, _use_new_zipfile_serialization=False)
```

## Run
### Train 
To train the model, run 
```
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} dist_train.py \
                                                      --data_dir ${DATA_DIR} \
                                                      --output_dir ${OUTPUT_DIR} \
                                                      [--epochs ${EPOCHS}] \
                                                      [--batch_size ${BATCH_SIZE}] \
                                                      [--learning_rate ${LEARNING_RATE}]] \
                                                      [--resume_from ${CHECK_POINT}] \
                                                      [--pretrained ${DIR}] \
                                                      [--channel_attention ${BOOL}] \
                                                      [--spatial_attention ${BOOL}] \
                                                      [--use_sigmoid ${BOOL}] 
```
To train the model with the discriminator, run 
```
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} dist_train_lr_metric.py \
                                                      --data_dir ${DATA_DIR} \
                                                      --output_dir ${OUTPUT_DIR} \ 
                                                      [--epochs ${EPOCHS}] \
                                                      [--batch_size ${BATCH_SIZE}] \
                                                      [--learning_rate ${LEARNING_RATE}]] \
                                                      [--resume_from ${CHECK_POINT}] \
                                                      [--pretrained ${DIR}] \
```
### Demo
We provide a GUI window for demostration
```
TO DO
```

## Reference
Our work is inspired by the papers below:
```
 @Article{IizukaSIGGRAPH2016,
   author = {Satoshi Iizuka and Edgar Simo-Serra and Hiroshi Ishikawa},
   title = {{Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for 
            Automatic Image Colorization with Simultaneous Classification}},
   journal = "ACM Transactions on Graphics (Proc. of SIGGRAPH 2016)",
   year = 2016,
   volume = 35,
   number = 4,
 }
```

```
@article{deepkoal2017,
  author          = {Federico Baldassarre, Diego Gonzalez-Morin, Lucas Rodes-Guirao},
  title           = {Deep-Koalarization: Image Colorization using CNNs and Inception-ResNet-v2},
  journal         = {ArXiv:1712.03400},
  url             = {https://arxiv.org/abs/1712.03400},
  year            = 2017,
  month           = dec
}
```
```
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  year      = {2015},
}
```
```
@article{DBLP:journals/corr/LarsenSW15,
  author    = {Anders Boesen Lindbo Larsen and Soren Kaae Sonderby and Ole Winther},
  title     = {Autoencoding beyond pixels using a learned similarity metric},
  journal   = {CoRR},
  year      = {2015},
}
```
