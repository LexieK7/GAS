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

Modify the code as needed (2.Quality/train.py)：
```
dataset = "DATASET PATH"
visual_model.load_state_dict(torch.load(os.path.join("UNI PATH", "pytorch_model.bin"), map_location="cuda"), strict=True)
```

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
