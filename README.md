# Convolution channel separation and frequency sub-bands aggregation for music genre classification

Pytorch code for following paper:

* **Title** : Convolution channel separation and frequency sub-bands aggregation for music genre classification (Announced at ICASSP2023, now available [here]( https://arxiv.org/abs/2211.01599 )) 
* **Autor** :  Jungwoo Heo, Hyun-seo Shin, Ju-ho kim, Chan-yeong Lim, and Ha-Jin Yu

# Abstract
In music, short-term features such as pitch and tempo constitute long-term semantic features such as melody and narrative. A music genre classification (MGC) system should be able to analyze these features. In this research, we propose a novel framework that can extract and aggregate both short- and long-term features hierarchically. Our framework is based on ECAPA-TDNN, where all the layers that extract short-term features are affected by the layers that extract long-term features because of the back-propagation training. To prevent the distortion of short-term features, we devised the convolution channel separation technique that separates short-term features from long-term feature extraction paths. To extract more diverse features from our framework, we incorporated the frequency sub-bands aggregation method, which divides the input spectrogram along frequency bandwidths and processes each segment. We evaluated our framework using the Melon Playlist dataset which is a large-scale dataset containing 600 times more data than GTZAN which is a widely used dataset in MGC studies. As the result, our framework achieved 70.4% accuracy, which was improved by 16.9% compared to a conventional framework.

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:22.01-py3' image of Nvidia GPU Cloud for conducting our experiments. 

* Python 3.8.12

* Pytorch 1.11.0+cu115

* Torchaudio 0.11.0+cu115

  

# Datasets

We used GTZAN dataset and Melon Playlist dataset. We partitioned the training and test sets following the k-fold (k==10) process. You can use our 10-fold information at [here](  )(GTZAN) and [here](  )(Melon Playlist).

> If you want to make new 10-fold information, just run `/GTZAN/k_fold.py` in GTZAN dataset or `Melon/k_fold.py` in Melon Playlist dataset.

# Run experiment

Go into experiment folder what you want to run.

### Set system arguments

First, you need to set system arguments. You can set arguments in `#00.Experiements/arguments.py`. Here is list of system arguments to set.

```python
1. 'path_log'	  : {YOUR_PATH}
	'path_log' is path of saving experiments.
	input type is str

	CAUTION! Don't set your path_log inside the experiment code path.
	'~/#00.Experiment_name/log_path' (X)
	'~/result/log_path'(O)

2. 'path_gtzan'  : {YOUR_PATH}
	'path_gtzan' is path where GTZAN dataset is saved.
	input type is str

	# ex) '/data/GTZAN'

3. 'kfold_ver'   : {K-FOLD_VER}
	'kfold_ver' is k number of k-Fold.
	You can use [1,2,3,4,5,6,7,8,9,10] numbers.
	input type is int

	# ex) 1
```

### Additional logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`

```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'Hyun-seo',
'wandb_token'  : 'WANDB_TOKEN',
```

2. In `main.py`

```python
# Just remove "#" in logger

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        ).use_wandb(args['wandb_user'], args['wandb_token']
        ).build()
```

### Run experiment code

```python
# And just run main.py
python main.py
```

We adopt 10-fold cross-validation to get system performance (accuracy). So you need to change `kfold_ver` and re-run experiment code.



# Citation

Please cite this paper if you make use of the code. 

```
@article{
}
```

# Reference
We implemented the BBNN system with reference to [here]( https://arxiv.org/pdf/1901.08928.pdf ). The original BBNN code can be found [here]( https://github.com/CaifengLiu/music-genre-classification ).
