# FIXME: please set system arguments
"""
README
list of system arguments to set
1. 'path_log'	  : {YOUR_PATH}
	'path_log' is path of saving experiments.
	input type is str

	CAUTION! Don't set your path_log inside the experiment code path.
	'~/#00.Experiment_name/log_path' (X)
	'~/result/log_path'(O)

2. 'path_melon'  : {YOUR_PATH}
	'path_melon' is path where Melon Playlist dataset is saved.
	input type is str

	# ex) '/data/melon'

3. 'melon_kfold_ver'   : {K-FOLD_VER}
	'melon_kfold_ver' is k number of k-Fold.
	You can use [1,2,3,4,5,6,7,8,9,10] numbers.
	input type is int

	# ex) 1

ADDITIONAL
If you want to use wandb, set 'wandb_user' and 'wandb_token'.
"""

import os
import itertools
import torch 

def get_args():
	"""
	Returns
		system_args (dict): path, log setting
		experiment_args (dict): hyper-parameters
		args (dict): system_args + experiment_args
	"""
	system_args = {
		# expeirment info
		'project'	    : 'MGC',
		'name'		    : 'BBNN_1',
		'tags'		    : ['BBNN_melon'],
		'description'   : '',

		# log
		'path_log'	    : {YOUR_PATH}, # ex) '/code/log_path' 
		#'wandb_user'	: '',
		#'wandb_token'  : '',
		
		# path
		'path_melon'		: {YOUR_PATH}, # ex) '/data/GTZAN'
        'melon_kfold_ver'   : {K-FOLD_VER}, # ex) 1

		# env
		'num_workers': 4,
		'usable_gpu': None,
	}

	experiment_args = {
		# general
		'rand_seed'	    : 3823,
		'epoch'		    : 80,
		'batch_size'	: 128,

		# data
		'C'				: 32,
		'crop_size'	    : 647,

		# optimizer
        'lr'            : 1e-2,
		'lr_min'        : 1e-5,
        'weight_decay'  : 1e-5,
	}

	args = {}
	for k, v in itertools.chain(system_args.items(), experiment_args.items()):
		args[k] = v
	args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))

	return args, system_args, experiment_args