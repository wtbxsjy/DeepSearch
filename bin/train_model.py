import logging
import os
import re
import sys
import random
import yaml

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from datetime import datetime

from DeepSearch.training.params import parse_args
from DeepSearch.training.distributed import *
from DeepSearch.model.models import * 
from DeepSearch.training.loss import CocaLoss
from DeepSearch.training.dataset import *
from DeepSearch.training.sampler import *
from DeepSearch.training.scheduler import *
from DeepSearch.utils.tokenizer import *
from DeepSearch.training.training import evaluate, train_one_epoch, cal_database_emd
#from torch.distributed.elastic.multiprocessing.errors import record 
import pickle



def pt_load(file_path, map_location=None):
    with open(file_path, 'rb') as f:
        out = torch.load(f, map_location=map_location)
    return out


def setup_logger(log_file, level):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)    


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    


def main(args):
    seed = 89
    set_seed(89)

    args = parse_args(args)
    device = init_distributed_device(args)

    # setup experiment name
    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            date_str = broadcast_object(args, date_str)

        args.name = '-'.join([
            f"CoCa",
            date_str,
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
        ])
    

    # set up logger
    args.log_path = None 
    log_base_path = os.path.join(args.log_dir, args.name)
    if is_master(args):
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, 'out.log')
    
    args.log_level = logging.INFO
    setup_logger(args.log_path, args.log_level)

    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        logging.info(f'Running experiment {args.name}')
        args.tensorboard_path = os.path.join(
            log_base_path, "tensorboard")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
    
    # resume from checkpoint?
    if args.resume is not None:
        if is_master(args):
            if os.path.exists(args.resume):
                logging.info(f'Resuming from checkpoint at {args.resume}.')
            else:
                print('Error, can not find checkpoint at ' + args.resume)
                return -1
        if args.distributed:
            args.resume = broadcast_object(args, args.resume)

    
    if args.distributed:
        logging.info(
            f'Distributed running. Device: {args.device}. Process (global: {args.rank}, local{args.local_rank}), total {args.world_size}'
        )
    else:
        logging.info(
            f'Running with a single process. Device: {args.device}.'
        )
    

    config = None
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    bias_max_charge = config['Meta']['max_charge']
    if 'bias_max_charge' in config['Meta']:
        bias_max_charge = config['Meta']['bias_max_charge']
    bias_dim = bias_max_charge * 6
    # model, optimizer, scaler
    model = DeepSearch(**config["Model"]["params"],
                 bias_dim=bias_dim).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True)
    #model = torch.compile(model)
    if args.train_data:
        optimizer = optim.AdamW(model.parameters(), 
                                lr=args.lr)
        scaler = GradScaler()
    loss = CocaLoss(rank=args.rank, world_size=args.world_size)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, 'cpu')
        start_epoch = checkpoint["epoch"]
        state_dict = checkpoint["state_dict"]
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

        logging.info(f"Resuming from checkpoint {args.resume} epoch {start_epoch}")
    



    # dataset
    train_set = SpectrumDataset(
        args.train_data, config["Meta"], augmentation_prob=args.augmentation, masking_prob=args.augmentation_mask)
    
    batch_size = args.batch_size * args.world_size * args.accum_freq
    if args.mass_anchored:
        logging.info(f"Using mass anchored sampler")
        train_sampler = UniqueMassAnchoredContrastivePSMSampler(
                train_set, batch_size)
    else:
        train_sampler = ContrastivePSMSampler(train_set, batch_size)
    train_loader = DataLoader(train_set,
                              num_workers=args.n_workers,
                              persistent_workers=True,
                              pin_memory=True,
                              batch_sampler=train_sampler)
    
    
    val_set = SpectrumDataset(
        args.val_data, config["Meta"], augmentation_prob=0)
    
    val_sampler = UniqueMassAnchoredContrastivePSMSampler(val_set, args.batch_size * 2)
    val_loader = DataLoader(val_set, 
                            batch_sampler=val_sampler,
                            num_workers=args.n_workers, 
                            persistent_workers=True,
                            pin_memory=True)
    
    # scheduler
    total_steps = len(train_loader) * args.n_epochs
    #schedular = linear_lr_decay(
    #    optimizer, args.lr, total_steps, 2000, ln_rate_end)
    schedular = cosine_lr(optimizer, args.lr, warmup_length=2000, total_steps=total_steps)
    # tensorboard
    tb_writer = None
    if is_master(args):
        tb_writer = tensorboard.SummaryWriter(args.tensorboard_path)

    tokenizer = Tokenizer()
    # start training
    for epoch in range(start_epoch, args.n_epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        
        train_one_epoch(
            model=model,
            epoch_index=epoch,
            train_loader=train_loader,
            train_sampler=train_sampler,
            loss_fn=loss,
            optimizer=optimizer,
            accum_freq=args.accum_freq,
            scaler=scaler,
            scheduler=schedular,
            device=device,
            args=args,
            tb_writer=tb_writer,
            tokenizer=tokenizer,
        )
            

        completed_epoch = epoch + 1
        if is_master(args):
            evaluate(model=model,
                     val_set=val_set,
                     val_loader=val_loader,
                     val_sampler=val_sampler,
                     epoch=completed_epoch,
                     device=device,
                     args=args,
                     tb_writer=tb_writer)
            
        if is_master(args):
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scalar": scaler.state_dict()
            }
            if completed_epoch == args.n_epochs or (completed_epoch % 1) == 0:
                torch.save(checkpoint_dict,
                           os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))



if __name__ == "__main__":
    main(sys.argv[1:])
    
