import wandb
import torch
import numpy as np
import os, sys
import logging
import copy
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score

from args import get_copur_args
from Models.autoencoder import CoPurAE

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Utils.general import get_num_classes
from Utils.metrics import Meter

def _evaluate(model, test_loader, criterion, device):
    model.eval()
    model.to(device)

    losses = Meter(ptag='Loss')
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            output = model(data)

            loss = criterion(output, data)
            losses.update(loss.item(), data.size(0))
        
    return losses.avg

def _train_and_evaluate(model, training_criterion, optimizer, scheduler, train_loader, test_loader, 
                        iterations, device, test_every=5, use_wandb=False):

    model.train()
    model = model.to(device)
    best_model = None

    epoch = 0
    best_loss = 1e8

    while epoch < iterations:
        epoch += 1
        model.train()

        losses = Meter(ptag='Loss')

        for batch_idx, (data, target) in enumerate(train_loader):
            # data loading for GPU
            data = data.to(device)

            # forward pass
            output = model(data)
            loss = training_criterion(output, data)

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            
            losses.update(loss.item(), data.size(0))

        scheduler.step()

        logging.debug('Epoch {ep} Loss {loss.avg:.5f}'.format(ep=epoch, loss=losses))
        
        if use_wandb:
            wandb.log({
                'train/loss': losses.avg,
            }, step=epoch)

        if epoch % test_every == 0:
            test_loss = _evaluate(model, test_loader, training_criterion, device)
            if test_loss < best_loss:
                best_loss = test_loss
                # store the best model by making a copy
                best_model = copy.deepcopy(model)
            logging.info('Epoch {ep} Test Loss {acc:.5f}'.format(ep=epoch, acc=test_loss))
            print('Epoch {} Test Loss {:.5f}'.format(epoch, test_loss))

            if use_wandb:
                wandb.log({
                    'test/loss': test_loss,
                }, step=epoch)

    return best_loss, best_model

if __name__ == "__main__":

    args = get_copur_args()

    # save all params to params.txt
    with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
        str_args = '\n'.join([f'{k}: {v}' for k, v in vars(args).items()])
        f.write(str_args)
    
    if args.wandb:
        # initialize wandb
        wandb.init(
            config=args,
            project=args.wandb_project, 
            entity=args.wandb_entity,
            name=args.save_dir.split('/')[-1],
        )
    
    # configure logging
    logging.basicConfig(
        filename='{}/log.txt'.format(args.save_dir),
        format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s", 
        level=logging.DEBUG if args.debug else logging.INFO,
        force=True  
    )

    # make training deterministic
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_classes = get_num_classes(args.dataset)
    n_clients = args.size
    rng = np.random.default_rng(seed=args.seed)

    trainset = trainset = torch.load(os.path.join(args.datapath, 'logit_trainset.pth'), map_location='cpu')
    if isinstance(trainset, dict):
        all_trainsets = []
        for i in range(n_clients):  all_trainsets += trainset[i]
        trainset = all_trainsets

    testset = torch.load(os.path.join(args.datapath, 'logit_testset.pth'), map_location='cpu')

    # evaluate accuracy of averaging
    averaged_logits = [(x.reshape(-1, n_classes).mean(dim=0).squeeze(), y) for x, y in testset] 
    y_preds = [torch.argmax(x) for x, _ in averaged_logits]
    y_trues = [y for _, y in averaged_logits]
    acc = accuracy_score(y_trues, y_preds)
    logging.info('Averaged logits accuracy: {:.3f}'.format(acc))

    in_out_dim = n_classes * n_clients
    
    model = CoPurAE(in_out_dim, in_out_dim, args.hidden_dim, args.encode_dim)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')
    
    training_criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    device = 'cuda' if args.gpu else 'cpu'

    best_loss, best_model = _train_and_evaluate(model, training_criterion, optimizer, scheduler, 
                        train_loader, test_loader, args.epochs, 
                        device, test_every=5, use_wandb=args.wandb)
    
    logging.info('Best loss: {:.3f}'.format(best_loss))
    
    # save the best model
    logging.info('==> Saving model')
    save_path = os.path.join(args.save_dir, 'model.pth')
    torch.save(best_model.state_dict(), save_path)

    if args.wandb: wandb.finish()