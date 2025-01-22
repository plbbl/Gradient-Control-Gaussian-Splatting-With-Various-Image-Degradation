# Gradient-Control-Gaussian-Splatting-With-Various-Image-Degradation

This repository contains the code and pretrained models for the paper **Gradient-Controlled Gaussian Splatting with Various Image Degradation**. 

## Table of Contents

- [Installation](#installation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Testing](#testing)

## Installation


1. Clone the repository:

    ```bash
    git clone https://github.com/plbbl/Gradient-Control-Gaussian-Splatting-With-Various-Image-Degradation.git
    cd Gradient-Control-Gaussian-Splatting-With-Various-Image-Degradation
    ```
   

## 2D Pretrained Models

Here, we provide the links to the pretrained models for various 2D models used in our paper. The majority of these pretrained models are provided by the original authors of the corresponding models.

- [Uformer](https://github.com/ZhendongWang6/Uformer)
- [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR)
- [PDD](https://github.com/Yuehan717/PDD?tab=readme-ov-file)
- [LightenDiffusion](https://github.com/JianghaiSCU/LightenDiffusion?tab=readme-ov-file)
- [Retinexformer](https://github.com/caiyuanhao1998/retinexformer?tab=readme-ov-file)




## Training

To train the model on your own dataset, follow these steps:

1. Prepare the images restored by the 2D restoration model and use colmap to extract their point clouds.

2. According to your requirements, modify the script/train.sh file.

3. Run the training script:

    ```bash
    bash script/train.sh
    ```



## Testing

After training, you can evaluate the model using the testing script.


1. The `train.sh` file already includes the corresponding rendering code and test code. Please modify it according to your needs.

    ```bash
    bash script/train.sh
    ```



