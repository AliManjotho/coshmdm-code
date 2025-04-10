# CoShMDM: Contact and Shape-Aware Latent Motion Diffusion Model for Human Interaction Generation
[Ali Asghar Manjotho](https://github.com/AliManjotho), Tekie Tsegay Tewolde, Ramadhani Ally Duma, Zhendong Niu.

The official PyTorch implementation of the paper [**"CoShMDM: Contact and Shape Aware Latent Motion Diffusion Model for Human Interaction Generation"**](https://alimanjotho.github.io/coshmdm/).

Please visit our [**webpage**](https://alimanjotho.github.io/coshmdm/) for more details.




![teaser](./assets/model.png)



## Getting started

This code was tested on `Windows11 24H2` and requires:

* Python 3.8.0
* PyTorch 1.13.1+cu117
* conda3 or miniconda3

### 1. Setup FFMPEG
* Download ffmpeg from https://www.ffmpeg.org/download.html#build-windows
* Extract it in `C:\ffmpeg`.
* Add `C:\ffmpeg\bin` in `PATH` environment variable.


### 2. Setup miniconda environment
```shell
conda create -n coshmdm python==3.8.0
conda activate coshmdm
python -m spacy download en_core_web_sm
pip install -r requirements.txt
pip install trimesh h5py chumpy
```

* Download dependencies:

```bash
bash protos/smpl_files.sh
bash protos/glove.sh
bash protos/t2m_evaluators.sh
```



### 3. Get datasets

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

### 1. Download checkpoints and evaluation models
Run the shell script:

```shell
./prepare/download_pretrain_model.sh
./prepare/download_evaluation_model.sh
```
This will download coshmdm.ckpt under .\checkpoints\ and bert.ckpt under .\eval_model\.

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


## 4. Interaction Generation

#### 4.1 Generate from a single prompt

```shell
python -m tools.infer --text_prompt "In an intense boxing match, one is continuously punching while the other is defending and counterattacking." --num_repetitions 3
```

#### 4.2 Generate from test set prompts (prompts.txt)

```shell
python -m tools.infer --num_repetitions 5
```

#### 4.3 Generate from custom text file

```shell
ppython -m tools.infer --num_repetitions 3 --text_file ./assets/sample_prompts.txt
```

The results will be rendered and put in ./results/ directory.




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


### Rendering SMPL meshes  in Blender

* Download and install blender https://www.blender.org/download/.
* `{VER}` = your blender version, replace it accordingly.
* Blender>Preferences>Interface> Check Developer Options
* Add the following paths to PATH environment variable.
```shell
C:\Program Files\Blender Foundation\Blender {VER}
C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin
```
* Run CMD as Administrator and follow these commands:
```shell
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m ensurepip --upgrade
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install matplotlib --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install hydra-core --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install hydra_colorlog --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install shortuuid --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install omegaconf --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
"C:\Program Files\Blender Foundation\Blender {VER}\{VER}\python\bin\python.exe" -m pip install moviepy==1.0.3 --upgrade  --target="C:\Program Files\Blender Foundation\Blender {VER}\{VER}\scripts\modules"
```

* To create SMPL mesh per frame run:

```shell
python -m visualize.render_mesh --input_path ./results/In_an_intense_boxing_match,_one_is_continuously_/ --repetition_num 0
```

**This script outputs:**
* `p1_smpl_params.npy` and `p2_smpl_params.npy` - SMPL parameters (thetas, root translations, vertices and faces)
* `obj_rep###` - Mesh per frame in `.obj` format.


## Blender Addon for CoShMDM

![teaser](./assets/addon-1.png)
![teaser](./assets/addon-2.png)
![teaser](./assets/addon-3.png)



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
