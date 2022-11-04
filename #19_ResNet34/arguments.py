# FIXME: please set system arguments
"""
README
list of system arguments to set
1. 'path_log': {YOUR_PATH} (str)
	'path_log' is path of saving experiments.

	CAUTION! Don't set your path_log inside the experiment code path.
	'~/#00.Experiment_name/log_path' (X)
	'~/result/log_path'(O)

2. 'path_melon': {YOUR_PATH} (str)
	'path_melon' is path where Melon Playlist dataset is stored.

	# ex) '/data/GTZAN'

3. 'kfold_ver': {K-FOLD_VER} (int)
	'kfold_ver' is k number of k-Fold.
	You can use [1,2,3,4,5,6,7,8,9,10] numbers.

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
		'project'       : 'MGC',
		'name'          : 'MelonExperiemnt',
		'tags'          : ['Baseline'],
		'description'   : '',

		# log
		'path_log'      : {YOUR_PATH}, # ex) '/results'
		#'neptune_user'  : '',
		#'neptune_token' : '',

        # dataset
        'path_melon'        : {YOUR_PATH}, # ex) '/datas/melon'
        'melon_kfold_ver'   : {K-FOLD_VER}, # ex) 1

        # others
        'num_workers'   : 8
    }

    experiment_args = {
        # general
        'rand_seed'     : 4021,
        'epoch'         : 80,
        'batch_size'    : 256,

        # data
        'crop_size'     : 200,

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