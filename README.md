# IF-Font: Ideographic Description Sequence-Following Font Generation

<img width="96%" alt="teaser" src="data/assets/teaser.png">

By Xinping Chen, Xiao Ke\*, Wenzhong Guo

The official pytorch implementation of the paper `IF-Font: Ideographic Description Sequence-Following Font Generation`.

paper: [[Preprint](data/assets/preprint.pdf)] [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html)]  
code: [[Github](https://github.com/Stareven233/IF-Font)]

## Training
### Dependencies
- python 3.10
- pytorch 2.2.1
- lightning 2.2.1
- cuda 11.8

### Environment Setup
**Step -1**: Create a conda environment and install all packages (not suggested).
```bash
conda create -n iffont -f environment.yaml
conda activate iffont
```

**Step 1**: Create a conda environment and activate it.
```bash
conda create -n iffont python=3.10 -y
conda activate iffont
```

**Step 2**: Install Pytorch.
```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# or
conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

[pytorch.org previous-versions](https://pytorch.org/get-started/previous-versions/)

**Step 3**: Install other packages.
```bash
pip install -r requirements.txt
```

### Data Preparation
**Step 1**: Modify `iffont/data/valid_characters.py` to include the characters for training. For fairness, IF-Font only uses 3500 common characters and randomly divides them into two subsets. If more characters are used, allowing the seen characters to cover a greater number of components, the model performance can be improved.

**Step 2**: Place the font files to be used for training into separate subfolders under `data/fonts/`. Similarly, the more numerous and diverse the fonts are, the better the training performance will be.

After that, the file tree of `data/` should be:
```
.
├──assets
│
├──fonts
│   ├── train
│   │   ├── a.ttf
│   │   ├── b.ttf
│   │   └── ...
│   └── val
│       ├── c.ttf
│       ├── d.ttf
│       └── ...
│
└──raw_files
```

**Step 3**: Download the `f=8, VQ (Z=256, d=4)` version of vqgan from [CompVis's Model Zoo](https://github.com/CompVis/latent-diffusion/tree/main?tab=readme-ov-file#pretrained-autoencoding-models). The corresponding configuration file can be obtained from [here](https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/vq-f8-n256/config.yaml).

Place both in a subdirectory of `vqgan-logs/` (e.g., `vqgan_openImages_f8_n256`) so that its contents look as follows:
```
.
├──vqgan_openImages_f8_n256
│   ├── checkpoints
│   │   └── model.ckpt
│   └── configs
│       └── config.yaml
```

Choosing other versions of vqgan or other vector-quantized models is also possible, but modifications to the corresponding configuration (step 4) or model architecture (downsampling resolution or codebook size) might be necessary.

**Step 4**: Modify the configuration files under `iffont/config/` according to your actual situation.

**Step 5**: Run `iffont/data/datasets_h5.py` to create the dataset file in `.h5` format.
```bash
cd iffont
python data/datasets_h5.py
```

If `if_fonts.h5` is generated in the `data` directory, it indicates that the execution was successful.

### Running
After data preparation:

train:
```bash
python run.py fit -c config/base.yaml -c config/train.yaml --name=any
```

test/metric:
```bash
python run.py fit -c config/base.yaml -c config/train.yaml
python run.py test -c config/base.yaml -c config/train.yaml --ckpt_path=path/to/checkpoint/last.ckpt --data.dict_kwargs.test_set=ufuc
```

- `name`: The name of the checkpoint, which can be arbitrarily set.
- `ckpt_path`: The path where the checkpoint is stored.
- `data.dict_kwargs.test_set`: Specifies the character set to use for testing. **ufuc (default)**: **u**nseen **f**ont (data/fonts/val) and **u**nseen **c**haracter (iffont/data/valid_characters/val_ch); **sfsc**: **s**een **f**ont (data/fonts/train) and **s**een **c**haracter (iffont/data/valid_characters/train_ch); and so on...

## Inference
After training:

Please refer to `iffont/inference.ipynb`, complete the configuration in the first cell and execute it.

## Citation
```BibTeX
@inproceedings{chen2024iffont,
 author = {Chen, Xinping and Ke, Xiao and Guo, Wenzhong},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {14177--14199},
 publisher = {Curran Associates, Inc.},
 title = {IF-Font: Ideographic Description Sequence-Following Font Generation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/19ded4cfc36a7feb7fce975393d378fd-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```

## Contact
If you have any questions, please create a new issue or contact Stareven233@outlook.com.
