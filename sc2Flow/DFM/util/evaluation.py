from tqdm import tqdm
import torch
from .metrics import compute_metrics_fast,compute_r_squared
import numpy as np

def evaluate(solver, eval_dataloader, R2_only=False):
    nfe = 64
    step_size = 1 / nfe

    solver.model.eval()
    pred_list = []
    real_list = []
    condition_list = []

    for data_iter_step, (
            x_0,x_1,label
    ) in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader),desc='generating'):
        device = 'cuda'
        x_0 = x_0.to(device)
        x_1 = x_1.to(device)
        c_rep= label['c_rep'].to(device)
        dose=label['dose'].to(device)
        condition = label['condition']

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = solver.sample(x_init=x_0,
                        step_size=step_size,
                        verbose=False,
                        return_intermediates=True,
                        c_rep=c_rep,
                        dose=dose,
                        )

            for rel_exp,pre_exp,c in zip(x_1, result[1],condition): #result[0] == x_0
                real_list.append(rel_exp)
                pred_list.append(pre_exp) # x1 and result[1] have same condition
            condition_list += condition

        condition_set = list(set(condition_list))

    real_tensor = torch.stack(real_list, dim=0)
    pred_tensor = torch.stack(pred_list, dim=0)
    # import pickle
    # with open(f'./results/real_tensor.pkl', 'wb') as f:
    #     pickle.dump(real_tensor, f)
    # with open(f'./results/pred_tensor.pkl', 'wb') as f:
    #     pickle.dump(pred_tensor, f)

    real_tensor = real_tensor.to(torch.float32).cpu()
    pred_tensor = pred_tensor.to(torch.float32).cpu()
    result = {}
    for c in tqdm(condition_set,desc='compute metrics'):
        eval_idxs = np.where(np.array(condition_list)==c)[0]
        real = real_tensor[eval_idxs]
        pred = pred_tensor[eval_idxs]
        if R2_only:
            result[c] = compute_r_squared(real, pred)
        else:
            result[c] = compute_metrics_fast(real,pred)

    #calcuate mean

    if R2_only:
        for c in result.keys():
            result[c] = [torch.tensor(result[c]).mean()]
        print(result)
        print('mean R2: ',torch.tensor(list(result.values())).mean())
        return torch.tensor(list(result.values())).mean()

    test_logs = {}
    for c,metrics in result.items():
        for m,v in metrics.items():
            key = f'test_{m}_mean'
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(v)

    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]

    import pickle
    # save test_logs
    with open(f'./results/test_logs_full.pkl', 'wb') as f:
        pickle.dump(test_logs, f)

    for metric in test_logs.keys():
        test_logs[metric] = [torch.tensor(test_logs[metric]).mean()]
        print(metric,': ',test_logs[metric])

    return test_logs
