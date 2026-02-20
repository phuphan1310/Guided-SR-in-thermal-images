<p align="center">Thermal Image Super-Resolution via Transfer Learning on Multi-Domain Datasets</p>
<p align="center">Adapted from SGNet for Multispectral Satellite and Aerial Imagery</p>
This repository contains our implementation of thermal image super-resolution based on the SGNet architecture, enhanced through sequential transfer learning across multiple datasets for improved generalization in remote sensing applications.
Dependencies
bash
Python==3.11.5
PyTorch==2.1.0
numpy==1.23.5 
torchvision==0.16.0
opencv-python==4.8.1
scipy==1.11.3
tqdm==4.65.0
Pillow==10.0.1
matplotlib==3.8.0
Datasets
FLIR ADAS

KAIST Multispectral

VEDAI

SugarBeet2016

Models
All pretrained models can be accessed upon request. Please note that our model undergoes sequential fine-tuning across four datasets, with progressive improvement in performance.

Training Strategy
Stage 1: Initial Training on FLIR
bash
python train_flir.py --scale 4 --num_feats 32 --epoch 200 --batchsize 4 --max_samples 1000
Stage 2: Fine-tuning on KAIST
bash
python train_kaist_from_flir.py --pretrained [FLIR_MODEL_PATH] --scale 4 --num_feats 32 --epoch 100 --lr 0.00005 --batchsize 4 --max_samples 1000
Stage 3: Fine-tuning on VEDAI
bash
python train_vedai_from_kaist.py --pretrained [KAIST_MODEL_PATH] --scale 4 --num_feats 32 --epoch 100 --lr 0.00005 --batchsize 4 --max_samples 1000
Stage 4: Final Fine-tuning on SugarBeet2016
bash
python train_sugarbeet_from_vedai.py --pretrained [VEDAI_MODEL_PATH] --scale 4 --num_feats 32 --epoch 100 --lr 0.00005 --batchsize 4 --max_samples 1000
Test
Test on any dataset
bash
python test_model.py --model [MODEL_PATH] --rgb [RGB_IMAGE] --depth [DEPTH_IMAGE] --gt [GROUND_TRUTH] --scale 4 --target_size [WIDTH HEIGHT]
Experimental Results
Training Stage	Dataset	RMSE
Initial	FLIR	0.0499
+ Fine-tune	KAIST	0.0095
+ Fine-tune	VEDAI	0.0086
+ Fine-tune	SugarBeet2016	<0.008
Visual Comparison
<p align="center"> <img src="figs/comparison.png" width="800"/> </p> <p align="center">Comparison of thermal image super-resolution results across different training stages.</p>
Application: Precision Agriculture Monitoring
We integrated the enhanced thermal imagery with a YOLO-based segmentation pipeline for plant health assessment:

RGB Segmentation: YOLO generates precise leaf masks at individual plant level

Thermal Enhancement: SGNet upsamples low-resolution thermal imagery guided by RGB

Multi-modal Fusion: Thermal data overlaid with segmentation masks for per-plant temperature extraction

Anomaly Detection: Automatic identification of vegetation with thermal patterns indicating water stress

Reporting: Bounding boxes, GPS coordinates, and annotated imagery transmitted to central server

<p align="center"> <img src="figs/pipeline.png" width="900"/> </p> <p align="center">End-to-end pipeline for plant stress detection using RGB and thermal imagery.</p>
Inference on Edge Devices
Platform	Precision	FPS
Jetson Orin 8GB	FP16	40-50
Jetson Orin 8GB	FP32	20-30
RTX 4080 Laptop	FP16	100+
RTX 4080 Laptop	FP32	60-80
Acknowledgements
We thank the authors of SGNet for releasing their code and pretrained models. This work builds upon their excellent research and adapts it for multi-domain thermal image super-resolution applications.

Citation
If you find this work useful, please consider citing the original SGNet paper:

bibtex
@inproceedings{wang2024sgnet,
  title={Sgnet: Structure guided network via gradient-frequency awareness for depth map super-resolution},
  author={Wang, Zhengxue and Yan, Zhiqiang and Yang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5823--5831},
  year={2024}
}