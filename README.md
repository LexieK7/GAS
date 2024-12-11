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
#### The paths in the code need to change according to the actual requirements
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
### 2. Deployment of GAS Platform
#### [!]The platform is deployed in WINDOWS
Download [jdk](https://www.oracle.com/cn/java/technologies/downloads/#java8) and install it. 
Download the [pre-trained model](https://huggingface.co/wisdomik/QuiltNet-B-32/tree/main) and place them in the 3.GAS platform folder.

[CMD]Go to the directory where the folder resides.
```
java -jar sliceViewGadget.jar
```
[CMD]



## Citation
If you use this code for your research, please cite our paper.
```

```

If you use the original CUT model included in this repo, please cite the following papers.
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}

@misc{ikezogwo2023quilt1m,
      title={Quilt-1M: One Million Image-Text Pairs for Histopathology}, 
      author={Wisdom Oluchi Ikezogwo and Mehmet Saygin Seyfioglu and Fatemeh Ghezloo and Dylan Stefan Chan Geva and Fatwir Sheikh Mohammed and Pavan Kumar Anand and Ranjay Krishna and Linda Shapiro},
      year={2023},
      eprint={2306.11207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```
