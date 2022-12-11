<div align="center">

# Improving Transferability for Domain Adaptive Detection Transformers
  
  Kaixiong Gong, Shuang Li, Shugang Li, Rui Zhang, Chi Harold Liu, Qiang Chen
  
  ACM Multimedia 2022, Lisbon, Portugal.
  
  [![Paper](https://img.shields.io/badge/paper-arxiv.2208.01195-B31B1B.svg)](https://arxiv.org/abs/2204.14195) [[Supp](https://github.com/BIT-DA/O2net/blob/main/O2Net_supp.pdf)]
  
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-transferability-for-domain-adaptive/unsupervised-domain-adaptation-on-cityscapes-1)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-cityscapes-1?p=improving-transferability-for-domain-adaptive)

  
</div>

This repository contains the code of our ACM MM 2022 work "Improving Transferability for Domain Adaptive Detection Transformers".

If you found our paper or this project is useful to you, please cite:

```
@article{gong2022improving,
  title={Improving Transferability for Domain Adaptive Detection Transformers},
  author={Gong, Kaixiong and Li, Shuang and Li, Shugang and Zhang, Rui and Liu, Chi Harold and Chen, Qiang},
  journal={arXiv preprint arXiv:2204.14195},
  year={2022}
}
```

## Getting Started

### Prepare data

The second step is to prepare datasets. Four datasets are involved:

- Cityscapes [[download](https://www.cityscapes-dataset.com/downloads/)] (There are a lot of versions of Cityscapes, make sure you download the right one)
- Foggy Cityscapes [[downlaod](https://www.cityscapes-dataset.com/downloads/)]
- BDD100k [[download](https://doc.bdd100k.com/download.html#k-images)]
- Sim10k [[download](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)]

**Processing dataset annotations**. All annotations are should in COCO format which can be processed by the code. We should covert the above four datasets in to CoCo Format.

- Cityscapes to CoCo Format: using this [script](https://github.com/BIT-DA/O2net/blob/main/dataset_util/city2coco.py). (You can also refer to this [project](https://github.com/facebookresearch/maskrcnn-benchmark/tree/main/maskrcnn_benchmark/data#creating-symlinks-for-cityscapes))
- BDD100k to CoCo Format: using this [script](https://github.com/BIT-DA/O2net/blob/main/dataset_util/bdd2coco.py).
- Sim10k to CoCo Format: using this [script](https://github.com/BIT-DA/O2net/blob/main/dataset_util/sim2coco.py).

We also prived processed [data lists](https://drive.google.com/drive/folders/1aqneAxjGH0hfx9cBpBll0vDycfnHaR1w?usp=sharing) in CoCo Format.
- Cityscapes and Foggy Cityscapes default lists
- Cityscapes car only lists (for Sim10k to Cityscapes)
- BDD100k data lists
- Sim10k data list


### Requirements

See [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR#installation) for more details.

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

## Training

To train our method O2net on Cityscapes to Foggy Cityscapes, first pre-train a source model:

```
CUDA_VISIBLE_DEVICES=* GPUS_PER_NODE=n ./tools/run_dist_launch.sh n ./configs/r50_deformable_detr.sh --output_dir exps/source_model --dataset_file city2foggy_source
```

then:

```
CUDA_VISIBLE_DEVICES=* GPUS_PER_NODE=n ./tools/run_dist_launch.sh n ./configs/DA_r50_deformable_detr.sh --output_dir exps/ours --transform make_da_transforms --dataset_file city2foggy --checkpoint exps/source_model/checkpoint.pth
```

Or simply run:

```
sh DA.sh
```

## Testing

## Main results

|   Model   | Source Domain| Target Domain | mAP@50 |  Download    |
| --------- |:--------:|:-----------:|:-------------:|:-------------:|
| Source_only| Cityscapes  | Foggy Cityscapes |  28.6   | [Model](https://drive.google.com/file/d/1OD1y3j97fJgITvkqozJpDRyEtxuBKvU4/view?usp=sharing) |
| O2Net | Cityscapes  | Foggy Cityscapes | 47.2 | [Model](https://drive.google.com/file/d/1hv_w_hJF_rVgm77IO2Ct2JVi1Z4UfryD/view?usp=sharing) |

_Note_: The batch size is set as 2 on a single GPU.

## Acknowledgement

We thank the contributors of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR#installation) for their great work. We build our method on it.



