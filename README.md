# Colorization
This is the code of the colorization project of the National Innovation Program.
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

## reference
Our work is inspired by the papers below:
```
 @Article{IizukaSIGGRAPH2016,
   author = {Satoshi Iizuka and Edgar Simo-Serra and Hiroshi Ishikawa},
   title = {{Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification}},
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
