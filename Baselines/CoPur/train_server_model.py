import wandb
import torch
import numpy as np
import os, sys
import logging
import copy
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

from args import get_copur_args
from Models.server_model import CoPurAggregator
from Models.autoencoder import CoPurAE

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Utils.general import get_num_classes
from Utils.metrics import Meter, comp_accuracy

def do_ae_inference(ae_model, data_loader, device='cpu'):
    """
    Transform data through the autoencoder and return the decoder output.
    
    Args:
        ae_model: Pre-trained CoPur autoencoder model
        data_loader: DataLoader containing the data to transform
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        transformed_data, targets: Tuple of (transformed logits tensor, targets tensor)
    """
    ae_model.eval()
    ae_model.to(device)
    
    transformed_list = []
    targets_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            
            # Get decoder output from autoencoder
            reconstructed = ae_model(data)
            
            # Move back to CPU and store
            reconstructed = reconstructed.cpu()
            
            transformed_list.append(reconstructed)
            targets_list.append(target)
    
    # Concatenate all batches
    transformed_data = torch.cat(transformed_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    return transformed_data, targets

def create_transformed_dataloader(transformed_data, targets, batch_size, shuffle=False, num_workers=4):
    """
    Create a DataLoader from transformed data and targets.
    
    Args:
        transformed_data: Tensor of transformed logits
        targets: Tensor of targets
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader object
    """
    dataset = TensorDataset(transformed_data, targets)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

def _evaluate(model, test_loader, device, metric=None, require_argmax=True):
    model.eval()
    model.to(device)

    y_preds = []
    y_trues = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            outputs = model(data)

            y_trues.append(target.cpu())
            
            if require_argmax:
                y_preds.append(torch.argmax(outputs, dim=1).cpu())
            else:
                y_preds.append(outputs.cpu())

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)
    if metric is not None:
        acc1 = metric(y_trues_np, y_preds_np)
    else:
        acc1 = accuracy_score(y_trues_np, y_preds_np)
        
    return acc1

def _train_and_evaluate(model, training_criterion, optimizer, scheduler, train_loader, test_loader, 
                        iterations, device, test_every=5, metric=None, require_argmax=True, use_wandb=False):

    model.train()
    model = model.to(device)
    best_model = None

    epoch = 0
    best_acc = 0.0

    while epoch < iterations:
        epoch += 1
        model.train()

        losses = Meter(ptag='Loss')
        top1 = Meter(ptag='Prec@1') 

        for batch_idx, (data, target) in enumerate(train_loader):
            # data loading for GPU
            data = data.to(device)
            target = target.to(device)

            # forward pass
            output = model(data)
            loss = training_criterion(output, target)

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()

            # write log files
            train_acc = comp_accuracy(output, target)
            
            losses.update(loss.item(), data.size(0))
            top1.update(train_acc[0].item(), data.size(0))

        scheduler.step()

        logging.debug('Epoch {ep} Loss {loss.avg:.4f} Train Acc {top1.avg:.3f}'.format(
            ep=epoch, loss=losses, top1=top1))
        
        if use_wandb:
            wandb.log({
                'train/loss': losses.avg,
                'train/acc': top1.avg,
            }, step=epoch)

        if epoch % test_every == 0:
            test_acc = _evaluate(model, test_loader, device, metric=metric, require_argmax=require_argmax)
            if test_acc > best_acc:
                best_acc = test_acc
                # store the best model by making a copy
                best_model = copy.deepcopy(model)
            logging.info('Epoch {ep} Test Acc {acc:.3f}'.format(ep=epoch, acc=test_acc))
            print('Epoch {} Test Acc {:.3f}'.format(epoch, test_acc))

            if use_wandb:
                wandb.log({
                    'test/acc': test_acc,
                }, step=epoch)

    return best_acc, best_model

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

    in_dim = n_classes * n_clients
    out_dim = n_classes
    
    model = CoPurAggregator(in_dim, out_dim)
    
    ae_model = CoPurAE(in_dim, in_dim, args.hidden_dim, args.encode_dim)
    ae_model.load_state_dict(torch.load(os.path.join(args.modelpath, 'model.pth'), map_location='cpu'))
    
    device = 'cuda' if args.gpu else 'cpu'
    
    # Create temporary data loaders for AE inference
    temp_train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    temp_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logging.info('==> Transforming training data through autoencoder')
    trainset_transformed, train_targets = do_ae_inference(ae_model, temp_train_loader, device)
    
    logging.info('==> Transforming test data through autoencoder')
    testset_transformed, test_targets = do_ae_inference(ae_model, temp_test_loader, device)
    
    logging.info(f'Transformed training set size: {trainset_transformed.shape}')
    logging.info(f'Transformed test set size: {testset_transformed.shape}')

    # Create data loaders with transformed data using proper TensorDataset
    train_loader = create_transformed_dataloader(
        trainset_transformed, train_targets, 
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = create_transformed_dataloader(
        testset_transformed, test_targets, 
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')
    
    training_criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    best_acc, best_model = _train_and_evaluate(model, training_criterion, optimizer, scheduler, 
                        train_loader, test_loader, args.epochs, 
                        device, test_every=5, use_wandb=args.wandb)
    
    logging.info('Best acc: {:.3f}'.format(best_acc))
    
    # save the best model
    logging.info('==> Saving model')
    save_path = os.path.join(args.save_dir, 'model.pth')
    torch.save(best_model.state_dict(), save_path)

    if args.wandb: wandb.finish()