import torch

import gc
# flow_matching
from flow_matching.path import MixtureDiscreteProbPath, MixturePathGeneralizedKL_ASL
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

# visualization
import os
from torch.utils.data import DataLoader
from dataset import process_data
from model import MLP
from util.evaluation import evaluate
from util.config_tool import get_config


def main(args):
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('Using gpu')
    else:
        device = 'cpu'
        print('Using cpu.')

    torch.manual_seed(42)

    epochs = 600
    # training arguments
    lr = 0.001

    dropout = 0.2
    vocab_size = 2
    dims = 2000
    hidden_dim = 1024

    epsilon = 1e-3
    factor = 0.5
    if dataset_name == 'zhaoSims' or dataset_name == 'sciplex':
        save_freq = 10
        patience = 10
        lr_patience = 2
        batch_size = 256
        test_batch_size = 4096
        n_layers = 12
    elif dataset_name == 'mcfarland':
            save_freq = 10
            patience = 10
            lr_patience = 2
            batch_size = 256
            test_batch_size = 4096
            n_layers = 8
    else:
        n_layers = 64
        save_freq = 5
        patience = 5
        lr_patience =2
        batch_size = 4096
        test_batch_size = batch_size
    best_score = float('-inf')
    patience_counter = 0
    early_stop_threshold = 0.001


    if train:
        pre_dataset = process_data(args, type='train')
        train_loader = DataLoader(pre_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    pre_dataset_test = process_data(args, type='test')
    test_loader = DataLoader(pre_dataset_test, batch_size=test_batch_size, shuffle=False)
    gc.collect()

    # probability denoiser model init
    probability_denoiser = MLP(
        input_dim=vocab_size,
        time_dim=768,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        length=dims,
        dropout=dropout,
        condition_num=2).to(device)

    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    # init optimizer
    optim = torch.optim.Adam(probability_denoiser.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=factor, patience=lr_patience, min_lr=1e-6
    )

    loss_fn = MixturePathGeneralizedKL(path=path)
    if train:
        from tqdm import tqdm

        for epoch in range(epochs):
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (x_0, x_1, label) in bar:
                optim.zero_grad()
                x_0 = x_0.to(device)
                x_1 = x_1.to(device)
                labels = {
                    'c_rep': label['c_rep'].to(device),
                    'dose': label['dose'].to(device)
                }

                # sample time (user's responsibility)
                t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)

                # sample probability path
                path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

                # discrete flow matching generalized KL loss
                logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t, **labels)
                loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

                # optimizer step
                loss.backward()  # backward
                optim.step()  # update
                current_lr = optim.param_groups[0]['lr']
                # log loss

                bar.set_postfix_str(
                    f'|epoch {epoch:6d}/{epochs} | loss {loss.item():8.4f} | lr {current_lr:.6f} | p {patience} | lr :{optim.param_groups[0]["lr"]}')

            if (epoch+1) % save_freq == 0:
                probability_denoiser.eval()
                with torch.no_grad():

                    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
                    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path,
                                                        vocabulary_size=vocab_size)
                    r2 = evaluate(solver, test_loader, R2_only=True)

                if r2 > best_score + early_stop_threshold:
                    best_score = r2
                    patience_counter = 0
                    torch.save(probability_denoiser.state_dict(), f'{args.OUTPUT_DIR}/model_best.pt')
                    log_text = f'Training completed. Best R²: {r2:.4f} at epoch {epoch}'
                    with open(f'{args.OUTPUT_DIR}/training_log.txt', 'w', encoding='utf-8') as f:
                        f.write(log_text)
                    print(f"New best R²: {r2:.4f} at epoch {epoch}")
                    print(f'model saved to {args.OUTPUT_DIR}/model_last.pt')
                else:
                    patience_counter += 1
                    print(f"R² did not improve: {r2:.4f} at epoch {epoch}, patience:{patience_counter}/{patience}")

                scheduler.step(r2)

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}, best R²: {best_score:.4f}")
                    break

    probability_denoiser.load_state_dict(torch.load(f'{args.OUTPUT_DIR}/model_best.pt', weights_only=False))
    probability_denoiser.eval()
    with torch.no_grad():

        wrapped_probability_denoiser = WrappedModel(probability_denoiser)
        solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path,
                                            vocabulary_size=vocab_size)
        r2 = evaluate(solver, test_loader, R2_only=True)
        print(r2)

if __name__ == '__main__':

    for dataset_name in [
        'zhaoSims',
        #'mcfarland',
        #'sciplex'
    ]:
        args = get_config(dataset_name)
        k = args.k
        split_key_base = args.split_key
        DATA_PATH = args.DataPath
        train = True

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                return torch.softmax(self.model(x, t, **extras), dim=-1)

        for i in range(0,k):
            if k != 1:
                args.split_key = split_key_base + '_' + str(i)
            else:
                args.split_key = 'mode'
            args.OUTPUT_DIR = f'./output/{dataset_name}/{args.split_key}'
            os.makedirs(args.OUTPUT_DIR, exist_ok=True)
            if train and os.path.exists(f'{args.OUTPUT_DIR}/model_best.pt'):
                print(f'{args.OUTPUT_DIR}/model_best.pt exists, skip training')
                continue
            main(args)


