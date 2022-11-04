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

2. 'path_gtzan'  : {YOUR_PATH}
	'path_gtzan' is path where GTZAN dataset is saved.
	input type is str

	# ex) '/data/GTZAN'

3. 'kfold_ver'   : {K-FOLD_VER}
	'kfold_ver' is k number of k-Fold.
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
		'name'		    : 'ResNet34_gtzan_1',
		'tags'		    : ['SE-200'],
		'description'   : '',

		# log
		'path_log'	    : {YOUR_PATH}, # ex) '/code/log_path' 
		#'wandb_user'	: '',
		#'wandb_token'  : '',
		
		# path
		'path_gtzan'		: {YOUR_PATH}, # ex) '/data/GTZAN'
        'kfold_ver'         : {K-FOLD_VER}, # ex) 1
		
		# env
		'num_workers': 8,
		'usable_gpu': None,
	}

	experiment_args = {
		# general
		'rand_seed'	    : 3823,
		'epoch'		    : 80,
		'batch_size'	: 64,

		# data
		'crop_size'	    : 200,
        'data_cycle'    : 8,

		# optimizer
        'lr'            : 1e-3,
        'lr_min'        : 1e-6,
        'weight_decay'  : 1e-5,
	}


	args = {}
	for k, v in itertools.chain(system_args.items(), experiment_args.items()):
		args[k] = v
	args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))

	return args, system_args, experiment_args