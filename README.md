# GAS

## Getting started
Clone this repo:
```
https://github.com/LexieK7/GAS.git
cd GAS
```

### 1. Deployment of Generative Model


Modify the pretrained model (1.FS2FFPE/models/networks.py):
```
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="PATH OF QUILTNET WEIGHT")
```

Train the FS2FFPE model:
```
python train.py --dataroot ./datasets/FS2FFPE --name FS2FFPE --CUT_mode CUT
```

Test the FS2FFPE model:
```
python test.py --dataroot ./datasets/FS2FFPE --name FS2FFPE --CUT_mode CUT --phase test
```

### 2. Deployment of Quality Assessment Model

Train the Quality Assessment Model:
```
python train.py
```

Test the Quality Assessment Model:
```
python test.py
```

Visualization:
```
python visual.py
```

## Citation
If you use this code for your research, please cite our paper.
```

```

If you use the original CUT model included in this repo, please cite the following paper.
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
