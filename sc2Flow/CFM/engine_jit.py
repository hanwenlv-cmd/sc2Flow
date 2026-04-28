import math
import sys
import pickle
import torch
import numpy as np
import os
import util.misc as misc
import util.lr_sched as lr_sched

import copy
from utils import list2tensor

from tqdm import tqdm


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000
    loss_list = []
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    from tqdm import tqdm
    bar = tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header)),
                                            total=len(data_loader))
    for data_iter_step, (g_id, g_exp, c_rep, d, _) in bar:
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        labels = {'g_id':g_id.to(device),'c_rep':c_rep.to(device),'dose':d.to(device)}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(
                x=g_exp.to(device),
                labels=labels,
                step=data_iter_step
            )
        loss_value = loss.item()
        loss_list.append(loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        bar.set_postfix_str(f'{model.net.final_layer.linear.weight.grad.max():.4f}')
    return  sum(loss_list) / len(loss_list)

def evaluate(model_without_ddp, args, eval_dataloader):

    model_without_ddp.eval()

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    pred_list = []
    real_list = []
    condition_list = []
    for data_iter_step, (
            g_id, g_exp, c_rep, d, condition
    ) in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader),desc='generating'):
        device = 'cuda'

        labels_gen = {'g_id':g_id.to(device),
                      'c_rep':c_rep.to(device),
                      'dose':d.to(device),
                      'condition':condition}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = model_without_ddp.generate(labels_gen)
            for id,rel_exp,pre_exp,c in zip(g_id,g_exp,result,condition):
                full_rel_value = list2tensor(id.cpu(), rel_exp.cpu())
                real_list.append(full_rel_value)
                full_pre_value = list2tensor(id.cpu(), pre_exp.cpu())
                pred_list.append(full_pre_value)
            condition_list += condition

        condition_set = list(set(condition_list))

    real_tensor = torch.stack(real_list, dim=0)
    pred_tensor = torch.stack(pred_list, dim=0)

    with open(f'{args.output_dir}/real_tensor.pkl', 'wb') as f:
        pickle.dump(real_tensor, f)
    with open(f'{args.output_dir}/pred_tensor.pkl', 'wb') as f:
        pickle.dump(pred_tensor, f)

    from util.metrics import compute_metrics_fast

    def limit(x,n):
        if x.shape[0]>n:
            indices = torch.randperm(x.shape[0])[:n]
            sampled_tensor = torch.index_select(x, 0, indices)
            return sampled_tensor
        else:
            return x

    limit_n = 30000
    result = {}
    for c in tqdm(condition_set,desc='compute metrics'):
        eval_idxs = np.where(np.array(condition_list)==c)[0]
        real = real_tensor[eval_idxs]
        pred = pred_tensor[eval_idxs]

        real = limit(real.cpu(), limit_n).cuda()
        pred = limit(pred.cpu(), limit_n).cuda()
        result[c] = compute_metrics_fast(real,pred)

    #calcuate mean

    test_logs = {}
    for c,metrics in result.items():
        for m,v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(v)

    # save test_logs
    with open(f'{args.output_dir}/test_logs_full.pkl', 'wb') as f:
        pickle.dump(test_logs, f)

    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]
        print(metric,': ',test_logs[metric])




    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)


def limit(x,n):
    if x.shape[0]>n:
        indices = torch.randperm(x.shape[0])[:n]
        sampled_tensor = torch.index_select(x, 0, indices)
        return sampled_tensor
    else:
        return x


def evaluate_inference(model_without_ddp, args):
    import scanpy as sc
    from GeneDataset import gene_dataset,split_data
    from torch.utils.data import DataLoader
    import scipy

    model_without_ddp.eval()

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    adata_test_DFM = sc.read_h5ad(args.DFM_output_path)
    dataset = gene_dataset(adata_test_DFM,args, DFM_output_path=args.DFM_output_path)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)

    file_path = args.data_path
    print(f'reading {file_path}')
    adata = sc.read_h5ad(file_path)
    adata = split_data(adata, args, 'test')
    if os.path.exists(f'{args.output_dir}/pred_tensor_{args.num_sampling_steps}.pkl'):
        with open(f'{args.output_dir}/pred_tensor_{args.num_sampling_steps}.pkl', 'rb') as f:
            pred_tensor = pickle.load(f)
        print(f'load {args.output_dir}/pred_tensor_{args.num_sampling_steps}.pkl')
        pred_condition_list = list(adata_test_DFM.obs['cov_drug_dose_name'])
    else:
        pred_value_list = []
        pred_condition_list = []
        for data_iter_step, (
                g_id, _, c_rep, d, condition
        ) in tqdm(enumerate(dataloader),total=len(dataloader),desc='generating'):
            device = 'cuda'
            labels_gen = {'g_id':g_id.to(device),
                          'c_rep':c_rep.to(device),
                          'dose':d.to(device),
                          'condition':condition}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model_without_ddp.generate(labels_gen)
                for pre_id,pre_exp,c in zip(g_id,result,condition):
                    full_pre_value = list2tensor(pre_id.cpu(), pre_exp.cpu())
                    pred_value_list.append(full_pre_value)
                pred_condition_list += condition

        pred_tensor = torch.stack(pred_value_list, dim=0)

        with open(f'{args.output_dir}/pred_tensor_{args.num_sampling_steps}.pkl', 'wb') as f:
            pickle.dump(pred_tensor, f)


    if scipy.sparse.issparse(adata.X):
        real_tensor = torch.tensor(adata.X.toarray())
    else:
        real_tensor = torch.tensor(adata.X)
    real_condition_list = list(adata.obs['cov_drug_dose_name'])

    from util.metrics import compute_metrics_fast
    limit_n = 20000
    result = {}
    result_de = {}
    for c in tqdm(list(set(pred_condition_list)),desc='compute metrics'):
        pred_idxs = np.where(np.array(pred_condition_list)==c)[0]
        real_idxs = np.where(np.array(real_condition_list)==c)[0]
        real = real_tensor[real_idxs]
        pred = pred_tensor[pred_idxs]

        real = limit(real.cpu(), limit_n).cuda()
        pred = limit(pred.cpu(), limit_n).cuda()

        result[c] = compute_metrics_fast(real,pred)

        # use any cell type, cus they have same DEGs
        bool_de = adata.var_names.isin(np.array(adata.uns['all_DEGs'][c]))
        result_de[c] = compute_metrics_fast(real[:, bool_de], pred[:, bool_de])

    #calcuate mean
    # save test_logs
    with open(f'{args.output_dir}/res.pkl', 'wb') as f:
        pickle.dump(result, f)
    with open(f'{args.output_dir}/res_de.pkl', 'wb') as f:
        pickle.dump(result_de, f)


    test_logs = {}
    for c,metrics in result.items():
        for m,v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(v)

    # de
    test_de_logs = {}
    for c, metrics in result_de.items():
        for m, v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_de_logs:
                test_de_logs[key] = []
            test_de_logs[key].append(v)


    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]
        print(metric,': ',test_logs[metric])

    for metric in test_de_logs.keys():
        test_de_logs[metric] = [torch.tensor(test_de_logs[metric]).mean()]
        print(metric, ' de: ', test_de_logs[metric])

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)