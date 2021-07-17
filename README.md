# Fast Spatial-Temporal Transformer Network (FastSTTN)

![teaser](https://github.com/R-Escher/FastSTTN/blob/master/docs/teaser.png?raw=true)

### [Paper](link to paper) | [Video](link to video)


<!-- ---------------------------------------------- -->
## Citation
If you find our code helpful, please cite us:
```
@inproceedings{,
  author = {},
  title = {Fast Spatial-Temporal Transformer Network},
  booktitle = {},
  year = {2021}
}
```

<!-- ---------------------------------------------- -->
## Installation

Clone this repository and use conda environment to install all required Python packages:
```
conda env create -f environment.yml 
conda activate faststtn
```

<!-- ---------------------------------------------- -->
## Data preparation



<!-- ---------------------------------------------- -->
## Train this model

Adjust the hyperparameter as you like in the _configs/youtube-vos.json_ or _configs/davis.json_ and train the model with:

```
python train.py --config configs/youtube-vos.json
```
or
```
python train.py --config configs/davis.json
```


<!-- ---------------------------------------------- -->
## Test this model

Test the inference of our model with some video samples:
```
python test.py --video examples/some_original_video.mp4 --mask examples/some_segmentation_mask  --ckpt checkpoints/faststtn.pth
```

If you want to infer from a sequence of images, pass the _-s_ argument as:
```
python test.py -s --video examples/some_original_video.mp4 --mask examples/some_segmentation_mask  --ckpt checkpoints/faststtn.pth
```


<!-- ---------------------------------------------- -->











<!-- ---------------------------------------------- -->
<!-- ---------------------------------------------- -->
<!-- ---------------------------------------------- -->
<!-- ---------------------------------------------- -->
<!-- ---------------------------------------------- -->
<!-- ---------------------------------------------- -->














