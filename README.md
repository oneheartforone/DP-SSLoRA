# DP-SSLoRA
The offical repository for all algorithms and code for the "<b>DP-SSLoRA: A Privacy-Preserving Medical Classification Model Combining Differential Privacy with Self-supervised Low-rank Adaptation</b>".
 
## Usage

#### Setup

we can install packages using provided `environment.yaml`.

```shell
cd ./moco_pretraining/scripts
conda env create -f environment.yaml
conda activate py39_tc113
```

### Dataset

We employ four datasets for experiments.

<b> CheXpert (Pretrain dataset) </b>

https://stanfordmlgroup.github.io/competitions/chexpert/

<b> RSNA </b>

https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

<b> COVID-QU </b>

https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

<b> Chest X-Ray 15K </b>

https://www.kaggle.com/datasets/scipygaurav/15k-chest-xray-images-covid19
<br/>

***

#### Prepare datasets
Please refer to the description in the <font color=green>datasets section</font> of the paper and use `\moco_pretraining\scripts\split_small_dataset.py` to preprocess these datasets.

#### Pretrained Weights
The pretrained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12qIcnPfHUTzpYqRJrqAv4VULVxJAgmQf) and put in `\results\Pretrain_in_chexpert\all_none_tuningbatch128` directory.

* * *
### Training

The training logs will be saved in `\results` directory.

Important parameter settings are in the <font color=green>Experimental settings section</font> . Of course, you can run the following code directly.

#### <b>Pretraining</b>
If you want to train from scratch, run the following code.
```shell
cd ./moco_pretraining/moco

python main_moco.py -a resnet18 --lr 0.0001 --batch-size 128 --epochs 60 --world-size 1 --rank 0 --mlp --moco-t 0.2 --from-imagenet --multiprocessing-distributed --aug-setting chexpert --rotate 10 --maintain-ratio --dist-url tcp://localhost:8087 --dist-backend gloo --train_data ../data/MoCo-CXR-main/data/CheXpert-v1.0-small/train --exp-name r8w1n416 -j 0
```
Or use the <b>pre-training files</b> we provide.

[Google Drive](https://drive.google.com/drive/folders/12qIcnPfHUTzpYqRJrqAv4VULVxJAgmQf)

* * *

#### Dowmstream Training

```shell
cd moco_pretraining\lora_DP

# DP-SSLoRA
python main_lora_paper1_eval1.py -a resnet18 --epochs 50 --binary --workers 0 --gpu 0

# Parameter strategies
python main_lora_parametersearch.py -a resnet18 --epochs 50 --workers 0 --gpu 0

python main_lora_parametersearch2.py -a resnet18 --epochs 50 --workers 0 --gpu 0

# Validate in partial dataset
python main_lora_partial.py -a resnet18 --epochs 50 --workers 0 --gpu 0
```

## Citation

If you find this code useful, please cite in your research papers.



## References

[1] Irvin, Jeremy, et al. "Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.
<br/><br/>
[2] G.Shih, C.C. Wu, S. S. Halabi,M. D. Kohli,L.M. Prevedello,T. s. Cook, A.  Sharma,J. K. Amorosa,V. Arteaga,M. Galperin-Aizenberg，R.R.Gill,M. c. Godoy，s. Hobbs, J.Jeudy,A. Laroia, P. N. Shah,D. Vummidi, K. Yaddanapudi, A. Stein,Augmenting the na-tional institutes of health chestradiograph dataset withexpertannotations of possible pneu-monia, Radiology: Artificial In-telligence 1 (1)(2019) e180041, pMID:33937785. doi:10.1148/ryai.2019180041.
<br/><br/>
[3] Exploring the effect of imageenhancement techniquesoncovid-19 detection using chestx-ray images, Computers inBiology and Medicine 132(2021) 104319. doi:https://doi.org/10.1016/j.compbiomed.2021.104319.
<br/><br/>
[4] M.E. H. Chowdhury, T. Rah-man, A. Khandakar,R. Mazhar, M. A. Kadir, Z.B. Mahbub,K.R. Islam, M. s.Khan, A. Iqbal, N. A. Emadi,M. B.L.Reaz,M.T. Islam, Can ai helpin screening viral and covid-19pneumonia?, IEEEAccess 8 (2020) 132665-132676. doi:10.1109/Access.2020.3010287.




