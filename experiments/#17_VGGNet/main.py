import numpy as np
import random

import torch

import data
import train
import arguments
import model
from log.controller import LogModuleController

# get arguments
args, system_args, experiment_args = arguments.get_args()

# set reproducible
random.seed(args['rand_seed'])
np.random.seed(args['rand_seed'])
torch.manual_seed(args['rand_seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# trainer
mgc_trainer = train.MGCTrainer()
mgc_trainer.args = args

# logger
logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_neptune(args['neptune_user'], args['neptune_token']
        ).build()
logger.log_parameter(experiment_args)
mgc_trainer.logger = logger

# dataloader
temp = data.get_dataloaders(args)
mgc_trainer.train_loader, mgc_trainer.val_loader, mgc_trainer.eval_loader, mgc_trainer.genres = temp

# model
vggnet = model.VGGNet(args).cuda()
mgc_trainer.model = vggnet

# optimizer
mgc_trainer.optimizer = torch.optim.Adam(
    vggnet.parameters(),
    lr=args['lr'],
    weight_decay=args['weight_decay']
)

# lr_scheduler
mgc_trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
    mgc_trainer.optimizer,
    step_size=args['lr_step_size'],
    gamma=args['lr_gamma']
)

mgc_trainer.run()