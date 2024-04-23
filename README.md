# <p align=center>`Global-Local Context Network for Person Search`</p>

This repo is the official implementation of "[**Global-Local Context Network for Person Search**](https://arxiv.org/pdf/2112.02500.pdf)" (___ICASSP 2023___).

> **Authors:**
> [Jie Qin](https://scholar.google.com/citations?user=mhPGcuwAAAAJ),
> [Peng Zheng](https://scholar.google.com/citations?user=TZRzWOsAAAAJ),
> [Yichao Yan](https://scholar.google.com/citations?user=ZPHMMRkAAAAJ),
> [Quan Rong](https://ieeexplore.ieee.org/author/37086038231),
> [Xiaogang Cheng](https://scholar.google.se/citations?user=y6SrwJgAAAAJ), &
> [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ).

[[arXiv](https://arxiv.org/pdf/2112.02500.pdf)] [[code](https://github.com/ZhengPeng7/GLCNet)] [[stuff](https://drive.google.com/drive/folders/1wbq5jptOGxXDE0ze1tAMdcvXEaE1Wybt)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-context-network-for-person/person-search-on-cuhk-sysu)](https://paperswithcode.com/sota/person-search-on-cuhk-sysu?p=global-local-context-network-for-person) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-context-network-for-person/person-search-on-prw)](https://paperswithcode.com/sota/person-search-on-prw?p=global-local-context-network-for-person)

+ Abstract:

    Person search aims to jointly localize and identify a query person from natural, uncropped images, which has been actively studied over the past few years. In this paper, we delve into the rich context information globally and locally surrounding the target person, which we refer to as scene and group context, respectively. Unlike previous works that treat the two types of context individually, we exploit them in a unified global-local context network (GLCNet) with the intuitive aim of feature enhancement. Specifically, re-ID embeddings and context features are simultaneously learned in a multi-stage fashion, ultimately leading to enhanced, discriminative features for person search. We conduct the experiments on two person search benchmarks (i.e., CUHK-SYSU and PRW) as well as extend our approach to a more challenging setting (i.e., character search on MovieNet). Extensive experimental results demonstrate the consistent improvement of the proposed GLCNet over the state-of-the-art methods on all three datasets. Our source codes, pre-trained models, and the new dataset are publicly available at: [this https URL](https://github.com/ZhengPeng7/GLCNet).

+ Overall architecture of our GLCNet:

![arch](README.assets/GLCNet_arch_v2.svg)

### Performance

|                      Datasets                       | CUHK-SYSU | CUHK-SYSU |   PRW    |   PRW    |
| :-------------------------------------------------: | :-------: | :-------: | :------: | :------: |
|                     **Methods**                     |    mAP    |   top-1   |   mAP    |  top-1   |
| [OIM](https://github.com/serend1p1ty/person_search) |   75.5    |   78.7    |   21.3   |   49.4   |
|     [NAE+](https://github.com/DeanChan/NAE4PS)      |   92.1    |   92.9    |   44.0   |   81.1   |
|                        TCTS                         |   93.9    |   95.1    |   46.8   |   87.5   |
|   [AlignPS+](https://github.com/daodaofr/AlignPS)   |   94.0    |   94.5    |   46.1   |   82.1   |
|   [SeqNet](https://github.com/serend1p1ty/SeqNet)   |   93.8    |   94.6    |   46.7   |   83.4   |
|                     SeqNet+CBGM                     |   94.8    |   95.7    | **47.6** |   87.6   |
|                       GLCNet                        |   95.5    |   96.1    |   46.7   |   84.9   |
|                     GLCNet+CBGM                     | **95.8**  | **96.2**  | **47.8** | **87.8** |

+ Different gallery size on CUHK-SYSU:

<img src="README.assets/one-two_step.png" />

+ Qualitative Results:

    <img src="README.assets/qual_res_allInOne.svg" />

### Env

```
conda create -n glc python=3.8 -y && conda activate glc
pip install -r requirements.txt
```

### Data
Find all relevant data on [my google-drive folder](https://drive.google.com/drive/u/0/folders/1bEd_NAzdP6xCrldQBcKOHkCvkupNy0ix).  
Set the variable `SYS_HOME_DIR` in `defaults.py` to the root path of all projects. I always set the structure of file system in my machine as `SYS_HOME_DIR/codes/[ps/...], SYS_HOME_DIR/datasets/[ps/...], SYS_HOME_DIR/weights/[swin/pvt/...]`.

### Train

`sh ./run_${DATASET}.sh CUDA_DEVICE`

### Test
`sh ./test_${DATASET}.sh CUDA_DEVICE`

### Inference
Run the `demo.py` to make inference on given images. GLCNet runs at **10.3 fps** on a single Tesla V100 GPU with batch_size 3.

### Weights
You can download our well-trained models -- cuhk_957.pth and prw_469.pth from [my google-drive folder for GLCNet](https://drive.google.com/drive/folders/1wbq5jptOGxXDE0ze1tAMdcvXEaE1Wybt).

### MovieNet-CS
Download the whole MovieNet-PS dataset from our [google-drive](https://drive.google.com/file/d/1TKIzsUUo4zlNJFLT1_KzQCsL2zkOzwUJ/view?usp=drive_link) or [BaiduDisk](https://pan.baidu.com/s/1MXxbuEQ9F5Y220t-Kw6cJg?pwd=PSWD) (25.2GB, with frames and annotations).  
To extend person search framework to a more challenging setting, i.e., movies. We borrow the character detection and ID annotations from the [MovieNet](http://movienet.site/) dataset to organize MovieNet-CS, and set different levels of training set and different gallery size same as CUHK-SYSU. MovieNet-CS is saved exactly the same format and structure as CUHK-SYSU, which could be of great convenience to further research and experiments. BTW, you can also download all the movie frames in MovieNet on their official website.

If your network is unstable, you can also take a look at this [google-drive folder](https://drive.google.com/drive/folders/1kUr7v9_LUSSjW5PyNbGqaiM6peXNvbiU) to separately download the annotation files and subsets of the frames, i.e., `frames_CS-1.zip ~ frames_CS-6.zip` and combine them together.

### Acknowledgement

Thanks to the solid codebase from [SeqNet](https://github.com/serend1p1ty/SeqNet).

### Citation

```bibtex
@article{zheng2021glcnet,
  title={Global-local context network for person search},
  author={Zheng, Peng and Qin, Jie and Yan, Yichao and Liao, Shengcai and Ni, Bingbing and Cheng, Xiaogang and Shao, Ling},
  journal={arXiv preprint arXiv:2112.02500},
  volume={8},
  year={2021}
}

@inproceedings{qin2023movienet,
  title={MovieNet-PS: a large-scale person search dataset in the wild},
  author={Qin, Jie and Zheng, Peng and Yan, Yichao and Quan, Rong and Cheng, Xiaogang and Ni, Bingbing},
  booktitle=ICASSP,
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

