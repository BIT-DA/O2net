<div align="center">

# Improving Transferability for Domain Adaptive Detection Transformers
  
  Kaixiong Gong, Shuang Li, et al.
  
  ACM Multimedia 2022, Lisbon, Portugal.
  
  [![Paper](https://img.shields.io/badge/paper-arxiv.2208.01195-B31B1B.svg)](https://arxiv.org/abs/2204.14195)
  
</div>

This repository contains the code of our ACM MM 2022 work "Improving Transferability for Domain Adaptive Detection Transformers".

## Getting Started

The first thing is built the environment. We should follow the following insturction (see [Deformable DETR project](https://github.com/fundamentalvision/Deformable-DETR#installation) for more details). 

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

