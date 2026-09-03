import logging
import torch
import copy
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim import Adam
import os
import torch.nn.functional as F
from math import comb
import wandb

from Utils.metrics import Meter, comp_accuracy
from Utils.fl_args import get_args
from Utils.adversarial import adversarial_attack_batch_inplace, adversarial_attack_batch, \
    sia_attack_blackbox_batch, sia_attack_blackbox_collude_batch, Carlini_Wagner_loss
from Utils.general import get_model, get_num_classes

def wandb_init_metrics():
    metrics = ["train/acc", "train/loss", "test/acc"]
    for metric in metrics: wandb.define_metric(metric)

def _evaluate(model, test_loader, device, metric=None, require_argmax=True):
    model.eval()
    model.to(device)

    y_preds = []
    y_trues = []
    with torch.no_grad():
        for data, mask, target in test_loader:
            data = data.to(device)
            mask = mask.to(device)
            
            outputs = model(data, mask)

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

        for batch_idx, (data, mask, target) in enumerate(train_loader):
            # data loading for GPU
            data = data.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # forward pass
            output = model(data, mask)
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

def _train_and_evaluate_adv(model, training_criterion, adv_criterion, optimizer, scheduler, train_loader, test_loader, 
                            iterations, device, n_adv=1, n_iter=10, alpha=0.1, eps=0.1, test_every=5, metric=None, 
                            require_argmax=True, new_adversaries=True, attack_type='sia', black_box=False, collude=False, 
                            use_wandb=False):

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

        for batch_idx, (data, mask, target) in enumerate(train_loader):
            logging.debug(f'Epoch {epoch} Batch {batch_idx}/{len(train_loader)}')
            # Move data to GPU
            data = data.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # --- Generate adversarial examples for the current batch ---
            if attack_type == 'pgd':
                if black_box:
                    raise ValueError("Adversarial training with PGD can only be launched as white-box")

                if new_adversaries:
                    # This function returns x_adv with shape [batch_size, n_clients + n_adv, n_classes]
                    # where the last n_adv rows are the adversarial perturbations.
                    # used for testing
                    adv_data, adv_mask = adversarial_attack_batch(
                        x=data, mask=mask, y=target, f=model, device=device, n_adv=n_adv, loss_fn=adv_criterion, 
                        n_iter=n_iter, alpha=alpha, eps=eps
                    )
                else:
                    ### in place adversaries, sets masks = 0 as adversaries
                    adv_data, adv_mask = adversarial_attack_batch_inplace(
                        x=data, mask=mask, y=target, f=model, device=device, loss_fn=adv_criterion, 
                        n_iter=n_iter, alpha=alpha, eps=eps
                    )
            elif attack_type == 'sia':
                if black_box and collude:
                    if batch_idx == 0: 
                        logging.info('==> Black-box SIA with colluding adversaries')
                    adv_data, adv_mask = sia_attack_blackbox_collude_batch(
                        f=model, x=data, mask=mask, device=device
                    )
                elif black_box and not collude:
                    if batch_idx == 0: 
                        logging.info('==> Black-box SIA with non-colluding adversaries')
                    adv_data, adv_mask = sia_attack_blackbox_batch(
                        x=data, mask=mask, device=device
                    )
                else:  # white-box attack
                    raise NotImplementedError("SIA attack is not implemented for white-box training")

            # move to device
            adv_data = adv_data.to(device)
            adv_mask = adv_mask.to(device)

            # --- Forward Pass ---            
            output_adv = model(adv_data, adv_mask)
            loss = training_criterion(output_adv, target)

            # --- Backward Pass & Update ---
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics (using clean accuracy, but you can also monitor adversarial accuracy)
            train_acc = comp_accuracy(output_adv, target)
            losses.update(loss.item(), data.size(0))
            top1.update(train_acc[0].item(), data.size(0))

        # Step the learning rate scheduler if needed.
        scheduler.step()

        logging.info('Epoch {ep} Loss {loss.avg:.4f} Train Acc {top1.avg:.3f}'.format(
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
                best_model = copy.deepcopy(model)
            logging.info('Epoch {ep} Test Acc {acc:.3f}'.format(ep=epoch, acc=test_acc))

            if use_wandb:
                wandb.log({
                    'test/acc': test_acc,
                }, step=epoch)

    return best_acc, best_model

def generate_subsets(ds, rng, min_set_size=5, max_set_size=15, n_subsets=10):
    new_ds = []
    for x, y in ds:
        n_clients = x.shape[0]
        # create n_subsets subsets for each x
        subset_sizes = rng.integers(min_set_size, max_set_size+1, size=n_subsets) 
        for subset_size in subset_sizes:
            subset = rng.choice(n_clients, size=subset_size, replace=False)
            new_ds.append((x[subset], y))
    return new_ds

def generate_subsets_with_masks(ds, rng, n_clients, min_set_size=5, max_set_size=15, n_subsets=10):
    new_ds = []
    lambda_ = 0.3

    valid_sizes = np.arange(min_set_size, max_set_size + 1)
    binom_coeffs = np.array([comb(n_clients, k) for k in valid_sizes], dtype=np.float64)
    prob_dist = binom_coeffs / binom_coeffs.sum()

    uniform_dist = np.ones_like(prob_dist, dtype=np.float64) / len(prob_dist)
    prob_dist = (1 - lambda_) * prob_dist + lambda_ * uniform_dist

    logging.info(f"Probability distribution for the range [{min_set_size}, {max_set_size}] is: {prob_dist}")
    # print(f"Probability distribution for the range [{min_set_size}, {max_set_size}] is: {prob_dist}")

    for x, y in ds:
        # create n_subsets subsets for each x
        subset_sizes = rng.choice(valid_sizes, size=n_subsets, replace=True, p=prob_dist)

        for subset_size in subset_sizes:
            subset = rng.choice(n_clients, size=subset_size, replace=False)
            mask = torch.zeros(x.size(0))
            mask[subset] = 1
            new_ds.append((x, mask, y))
    
    return new_ds

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)  # Unpack inputs and labels
    
    # Determine max size for padding
    max_size = max(item.size(0) for item in inputs)
    
    padded_batch = []
    masks = []
    label_batch = torch.tensor(labels)  # Stack labels directly
    
    for item in inputs:
        padding_size = max_size - item.size(0)
        padded_item = F.pad(item, (0, 0, 0, padding_size))  # Pad along the set dimension
        padded_batch.append(padded_item)
        
        # Create a mask (1 for valid elements, 0 for padded elements)
        mask = torch.ones(item.size(0)).to(item.device)
        mask = F.pad(mask, (0, padding_size))
        masks.append(mask)
    
    # Stack tensors and masks
    padded_batch = torch.stack(padded_batch)
    masks = torch.stack(masks)
    
    return padded_batch, masks, label_batch

if __name__ == "__main__":
    
    args = get_args()

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

        wandb_init_metrics()

    # make training deterministic
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_classes = get_num_classes(args.dataset)
    n_clients = args.size
    rng = np.random.default_rng(seed=args.seed)
    training_criterion = torch.nn.CrossEntropyLoss()

    if args.loss_fn == 'cw': ## attack trainable models using Carlini-Wagner loss
        adv_criterion = lambda logits, labels: \
            Carlini_Wagner_loss(logits, labels, input_is_prob=False, confidence=args.cw_confidence)
        print('==> Using Carlini-Wagner loss')
    else:
        adv_criterion = F.cross_entropy
        print('==> Using cross-entropy loss for logits')

    f = get_model(args.model, n_clients, n_classes, args.trim_ratio, args.dim_hidden, args.lip_scale, args.n_adv)
    
    device = 'cuda' if args.gpu else 'cpu'

    if args.optimizer == 'Adam':
        optimizer = Adam(f.parameters(), lr=args.lr)
    elif args.optimizer == 'Sgd':
        optimizer = torch.optim.SGD(f.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')
    
    # configure logging
    logging.basicConfig(
        filename='{}/log.txt'.format(args.save_dir),
        format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s", 
        level=logging.DEBUG if args.debug else logging.INFO,
        force=True  
    )
    
    trainset = torch.load(os.path.join(args.datapath, 'logit_trainset.pth'), map_location='cpu')
    if isinstance(trainset, dict):
        all_trainsets = []
        for i in range(n_clients):  all_trainsets += trainset[i]
        trainset = all_trainsets
    trainset = [(x.reshape(-1, n_classes), y) for x, y in trainset]
    if args.normalize and args.normalization_type in ['simplex']:
        logging.info('==> Normalizing trainset')
        trainset = [(torch.softmax(x, dim=-1), y) for x, y in trainset]
        # check if normalization is correct
        for i in range(5): 
            logging.debug('Norm: {} \n Max: {} \n Min: {} \n Sum: {} \n \n -------- \n'.format(
                torch.norm(trainset[i][0], p=2, dim=-1).tolist()[:5], 
                torch.max(trainset[i][0], dim=-1).values.tolist()[:5], 
                torch.min(trainset[i][0], dim=-1).values.tolist()[:5],
                torch.sum(trainset[i][0], dim=-1).tolist()[:5]
            ))
    else: 
        raise ValueError('Normalization disabled or normalization type not supported')

    if args.add_subsets:
        trainset = generate_subsets_with_masks(trainset, rng, n_clients, args.min_set_size, args.max_set_size, args.n_subsets)
    else:
        trainset = generate_subsets_with_masks(trainset, rng, n_clients, n_clients, n_clients, 1) # just includes masks
    logging.info(f'Trainset size: {len(trainset)}')

    testset = torch.load(os.path.join(args.datapath, 'logit_testset.pth'), map_location='cpu')
    testset = [(x.reshape(-1, n_classes), y) for x, y in testset]
    if args.normalize and args.normalization_type in ['simplex']:
        logging.info('==> Normalizing testset')
        testset = [(torch.softmax(x, dim=-1), y) for x, y in testset]
        # check if normalization is correct
        for i in range(5): 
            logging.debug('Norm: {} \n Max: {} \n Min: {} \n Sum: {} \n \n -------- \n'.format(
                torch.norm(testset[i][0], p=2, dim=-1).tolist()[:5], 
                torch.max(testset[i][0], dim=-1).values.tolist()[:5], 
                torch.min(testset[i][0], dim=-1).values.tolist()[:5],
                torch.sum(testset[i][0], dim=-1).tolist()[:5]
            ))
    testset = generate_subsets_with_masks(testset, rng, n_clients, n_clients, n_clients, 1) # just includes masks
    logging.info(f'Testset size: {len(testset)}')

    # evaluate accuracy of averaging
    averaged_logits = [(x.mean(dim=0).squeeze(), y) for x, _, y in testset] 
    y_preds = [torch.argmax(x) for x, _ in averaged_logits]
    y_trues = [y for _, y in averaged_logits]
    acc = accuracy_score(y_trues, y_preds)
    logging.info('Averaged logits accuracy: {:.3f}'.format(acc))

    # create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    if args.adversarial:
        if not args.new_adversaries: logging.info('==> Using in-place adversarial attack')
        best_acc, best_model = _train_and_evaluate_adv(f, training_criterion, adv_criterion, optimizer, scheduler, train_loader, test_loader, 
                                                       args.epochs, device, n_adv=args.n_adv, n_iter=args.n_iter, alpha=0.1, eps=0.1, 
                                                       test_every=1, require_argmax=True, new_adversaries=args.new_adversaries, 
                                                       attack_type=args.attack_type, black_box=args.black_box, collude=args.collude, 
                                                       use_wandb=args.wandb)
    else:
        best_acc, best_model = _train_and_evaluate(f, training_criterion, optimizer, scheduler, train_loader, test_loader, 
                                      args.epochs, device, test_every=5, require_argmax=True, use_wandb=args.wandb)

    logging.info('Best accuracy: {:.3f}'.format(best_acc))
    
    # save the best model
    logging.info('==> Saving model')
    save_path = os.path.join(args.save_dir, 'model.pth')
    torch.save(best_model.state_dict(), save_path)

    if args.wandb: wandb.finish()
    