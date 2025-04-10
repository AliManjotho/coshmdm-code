# CoShMDM: Contact and Shape-Aware Latent Motion Diffusion Model for Human Interaction Generation
### [Project Page](https://alimanjotho.github.io/coshmdm/) | [Ali Asghar Manjotho](https://github.com/AliManjotho), Tekie Tsegay Tewolde, Ramadhani Ally Duma, Zhendong Niu.</br>


Official PyTorch implementation for [CoShMDM: Contact and Shape Aware Latent Motion Diffusion Model for Human Interaction Generation](https://alimanjotho.github.io/coshmdm/).


<p float="left">
  <img src="./assets/model.png" width="900" />
</p>




## InterHuman Dataset
<!-- ![interhuman](https://github.com/tr3e/InterGen/blob/main/interhuman.gif) -->

InterHuman is a comprehensive, large-scale 3D human interactive motion dataset encompassing a diverse range of 3D motions of two interactive people, each accompanied by natural language annotations.

<p float="left">
  <img src="./assets/interhuman.gif" width="900" />
</p>

It is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can access the dataset in our [webpage](https://tr3e.github.io/intergen-page/) with the google drive link for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made. The redistribution of the dataset is **prohibited**. 



## Getting started

This code was tested on `Windows 11 24H2` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

```shell
conda create --name coshmdm
conda activate coshmdm
pip install -r requirements.txt
```

### 2. Get data


Download the data from [webpage](https://tr3e.github.io/intergen-page/). And put them into ./data/.

#### Data Structure
```sh
<DATA-DIR>
./annots                //Natural language annotations where each file consisting of three sentences.
./motions               //Raw motion data standardized as SMPL which is similiar to AMASS.
./motions_processed     //Processed motion data with joint positions and rotations (6D representation) of SMPL 22 joints kinematic structure.
./split                 //Train-val-test split.
```


## Demo



### 1. Download the checkpoint
Run the shell script:

```shell
./prepare/download_pretrain_model.sh
```

### 2. Modify the configs
Modify config files ./configs/model.yaml and ./configs/infer.yaml


### 3. Modify the input file ./prompts.txt like:

```sh
In an intense boxing match, one is continuously punching while the other is defending and counterattacking.
With fiery passion two dancers entwine in Latin dance sublime.
Two fencers engage in a thrilling duel, their sabres clashing and sparking as they strive for victory.
The two are blaming each other and having an intense argument.
Two good friends jump in the same rhythm to celebrate.
Two people bow to each other.
Two people embrace each other.
...
```

### 4. Run
```shell
python tools/infer.py
```
The results will be rendered and put in ./results/


## Train


Modify config files ./configs/model.yaml ./configs/datasets.yaml and ./configs/train.yaml, and then run:

```shell
python tools/train.py
```


## Evaluation



### 1. Modify the configs
Modify config files ./configs/model.yaml and ./configs/datasets.yaml

### 2. Run
```shell
python tools/eval.py
```


## Application
<p float="left">
  <img src="./assets/trajectorycontrol.gif" width="900" />
</p>



## Citation

If you find our work useful in your research, please consider citing:

```
@article{manjotho2025coshmdm,
  title={CoShMDM: Contact and Shape-Aware Latent Motion Diffusion Model for Human Interaction Generation},
  author={Ali Asghar Manjotho, Tekie Tsegay Tewolde, Ramadhani Ally Duma, Zhendong Niu},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```




## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can **use and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.
