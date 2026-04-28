import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from GeneDataset import process_data
import gc

def get_args_parser():

    parser = argparse.ArgumentParser('sc2Flow', add_help=False)

    # architecture
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to the dataset')
    parser.add_argument('--split_key', required=True, type=str)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    parser.add_argument('--gpu', type=str)
    parser.add_argument('--SMILES_key', type=str)
    parser.add_argument('--perturbation_key', type=str)
    parser.add_argument('--dosage_key', type=str)
    parser.add_argument('--cell_type_key', type=str)
    parser.add_argument('--control_key', type=str)
    parser.add_argument('--test_key', type=str)
    parser.add_argument('--DFM_output_path', type=str, default=None, help='if not None, CFM input tokens from DFM prediction')
    parser.add_argument('--evaluate_last', default=False, help='evaluate after last epoch')
    parser.add_argument('--evaluate_only', default=False, help='inference for evaluation')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='val_loss',
                        help='Metric to monitor for early stopping')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.evaluate_only and args.DFM_output_path==None:
        dataset_train = process_data(args,type='train')
        gc.collect()

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            #drop_last=True,
            collate_fn=dataset_train.collate_fn
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model_without_ddp = model
    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    if args.DFM_output_path is not None:
        print('inference using DFM outputs')
        with torch.random.fork_rng():
            torch.manual_seed(seed)

            from engine_jit import evaluate_inference
            with torch.no_grad():
                evaluate_inference(model_without_ddp,
                                   args,
                         )
        return
    # Evaluate generation
    if args.evaluate_only:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            dataset_test = process_data(args, type='test')
            gc.collect()

            print(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                shuffle=False,
                batch_size=args.gen_bsz,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=dataset_test.collate_fn
            )

            with torch.no_grad():
                evaluate(model_without_ddp,
                         args,
                         eval_dataloader=data_loader_test
                         )
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_metric = float('inf')
    patience_counter = 0
    early_stop = False

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        loss = train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, args=args)
        if loss < best_metric - args.min_delta:
            print(f'new best loss:{loss}')
            best_metric = loss
            patience_counter = 0
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )
        else:
            patience_counter += 1
            print(f'loss {loss} did not decreased, patience:{patience_counter}')

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            early_stop = True

        if early_stop:
            break

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name=f"epoch_{epoch}"
            )

    if args.evaluate_last:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            dataset_test = process_data(args, type='test')
            gc.collect()

            print(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                shuffle=False,
                batch_size=args.gen_bsz,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=dataset_test.collate_fn
            )

            with torch.no_grad():
                evaluate(model_without_ddp,
                         args,
                         eval_dataloader=data_loader_test
                         )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    data_config = {
        'mcfarland':{
            'k':5,
            'split_key':'split',
            'SMILES_key':'SMILES',
            'perturbation_key':'condition',
            'DataPath':'/home/usr/sc2Flow_lab/datasets/mcfarland_2020_pre.h5ad',
            'dosage_key':'dose_val',
            'cell_type_key':'cell_type',
            'control_key':'control',
            'test_key':'ood'
        },
        'zhaoSims': {
            'k': 4,
            'split_key': 'split',
            'SMILES_key': 'SMILES',
            'perturbation_key': 'condition',
            'DataPath': '/home/usr/sc2Flow_lab/datasets/zhaoSims2021_pre.h5ad',
            'dosage_key': 'dose_val',
            'cell_type_key': 'cell_type',
            'control_key': 'control',
            'test_key': 'ood'
        },
        'sciplex':{
            'k': 1,
            'split_key':'mode',
            'SMILES_key':'SMILES',
            'perturbation_key':'product_name',
            'DataPath':'/home/usr/sc2Flow_lab/datasets/sciplex3_pre_2000.h5ad',
            'dosage_key':'dose_val',
            'cell_type_key': 'cell_type',
            'control_key':'is_control',
            'test_key':'test',
        }
    }
    for dataset_name in [
        'mcfarland',
        'zhaoSims',
        'sciplex'
        ]:

        config = data_config[dataset_name]
        k = config['k']
        split_key_base = config['split_key']
        DATA_PATH = config['DataPath']
        for i in range(k):
            if k != 1:
                split_key = split_key_base + '_' + str(i)
            else:
                split_key = split_key_base
            OUTPUT_DIR = f'./output/{dataset_name}/{split_key}'
            #DFM_output_path = f'/home/usr/sc2FLow_lab/sc2Flow/DFM/output/{dataset_name}/{split_key}/predicted_adata.h5ad'
            args = get_args_parser().parse_args([
                "--proj_dropout", "0.0",
                "--P_mean", "-0.8",
                "--P_std", "0.8",
                "--noise_scale", "1.0",
                "--batch_size", "24",
                "--blr", "5e-5",
                "--epochs", "600",
                "--warmup_epochs", "5",
                "--gen_bsz", "128",
                "--cfg", "2.9",
                "--interval_min", "0.1",
                "--interval_max", "1.0",
                "--output_dir", OUTPUT_DIR,
                "--resume", OUTPUT_DIR,
                "--data_path", DATA_PATH,
                "--split_key", split_key,
                "--SMILES_key" , config['SMILES_key'],
                "--perturbation_key" , config['perturbation_key'],
                "--dosage_key" , config['dosage_key'],
                "--cell_type_key" , config['cell_type_key'],
                "--control_key", config['control_key'],
                "--test_key", config["test_key"],
                "--eval_freq",'599',
                '--save_last_freq','100',
                #"--DFM_output_path",DFM_output_path,
                #"--num_sampling_steps",'10', #5
                #"--evaluate_only",'True'
            ])

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            main(args)
