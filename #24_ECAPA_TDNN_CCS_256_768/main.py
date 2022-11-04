import os
import numpy as np
import random

import torch
import torch.nn as nn

import data
import train
import arguments
import model
from log.controller import LogModuleController

def set_experiment_environment(args):
    # reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'4021'
    args['rank'] = args['process_id']
    args['device'] = f'cuda:{args["process_id"]}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])

def run(process_id, args, experiment_args):
    # check parent process
    args['process_id'] = process_id
    args['flag_parent'] = process_id == 0

    # set reproducible
    set_experiment_environment(args)
    mgc_trainer = train.MGCTrainer()
    mgc_trainer.args = args

    # logger
    if args['flag_parent']:
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
    ecapa = model.ECAPA_TDNN(args).to(args['device'])
    ecapa = nn.SyncBatchNorm.convert_sync_batchnorm(ecapa)
    ecapa = nn.parallel.DistributedDataParallel(
        ecapa, device_ids=[args['device']], find_unused_parameters=True
    )
    mgc_trainer.model = ecapa

    # optimizer
    mgc_trainer.optimizer = torch.optim.Adam(
        ecapa.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    # lr_scheduler
    mgc_trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mgc_trainer.optimizer,
        T_max=args['epoch'],
        eta_min=args['lr_min']
    )

    mgc_trainer.run()

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # set reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])

    # set gpu device
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    
    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args,)
    )