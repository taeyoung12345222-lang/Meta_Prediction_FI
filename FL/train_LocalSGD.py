import os
import numpy as np
import time
import sys
from math import floor, ceil
import logging
import time
import datetime
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import pickle
from sklearn.metrics import accuracy_score
import wandb

from distoptim import FedProx, FedNova, FedAvg, FedAdam, FedYogi, LocalSGD, Scaffold
from stopping import check_stopping_criteria
import util_v4 as util
from args import get_args

args = get_args()

def maybe_barrier():
    # no-op when no process group was formed (see init_processes)
    if args.optimizer != 'localSGD':
        dist.barrier()

def run(rank, size):
    # initiate experiments folder
    save_path = args.savepath
    folder_name = os.path.join(save_path, args.name)
    if rank == 0 and os.path.isdir(folder_name)==False and args.save:
        os.mkdir(folder_name)
    maybe_barrier()

    # initiate log files
    logging.basicConfig(
        filename='{}/log_{}.txt'.format(folder_name, rank),
        format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s", 
        level=logging.INFO if not args.debug else logging.DEBUG,
        force=True  
    )
    tag_train = '{}/r{}_n{}_train.csv'
    tag_val = '{}/r{}_n{}_val.csv'
    tag_test = '{}/r{}_n{}_test.csv'
    params_file = '{}/params.dat'.format(folder_name)
    saveFileName_train = tag_train.format(folder_name, rank, size)
    saveFileName_val = tag_val.format(folder_name, rank, size)
    saveFileName_test = tag_test.format(folder_name, rank, size)
    args.out_fname = saveFileName_train
    args.out_fname_val = saveFileName_val
    args.out_fname_test = saveFileName_test
    args.out_fname_model = f'{folder_name}/{rank}_model.pth'
    args.out_fname_label_dist = f'{folder_name}/{rank}_label.pk'
    args.out_fname_agg_result = f'{folder_name}/{rank}_agg_result.json'

    # global validation accuracy (only for FL)
    if rank == 0 and args.rounds > 1:
        tag_global_val = '{}/r{}_n{}_global_val.csv'
        saveFileName_global_val = tag_global_val.format(folder_name, rank, size)
        args.out_fname_global_val = saveFileName_global_val
    
    with open(args.out_fname, 'w+') as f:
        print('Epoch,itr,Loss,avg:Loss,Prec@1,avg:Prec@1,ClientID',file=f)
    
    if(rank == 0 or args.rounds == 1):
        header = 'Epoch,Val,Time' if args.rounds > 1 else 'Epoch,Val'
        with open(args.out_fname_val, 'w+') as f:
            print(header,file=f)
        
        header = 'Epoch,Test,Time' if args.rounds > 1 else 'Epoch,Test'
        with open(args.out_fname_test, 'w+') as f:
            print(header,file=f)

    # Only rank 0 process logs test results
    if(rank == 0):
        with open(params_file, 'w+') as f:
            for k,v in vars(args).items():
               print(k, ':', v, file=f)

    # seed for reproducibility
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # gpu toggle
    if(args.gpu):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
      
    # load datasets
    dataset = util.get_dataset(args.totalclients, args.dataset, args)
    is_nlp = True if args.dataset in ['AG_News'] else False # used in train function
    total_clients = len(dataset.num_samples)
    metric = dataset.metric if hasattr(dataset, 'metric') else None
    require_argmax = dataset.require_argmax if hasattr(dataset, 'require_argmax') else True
    
    # define neural nets model, criterion, and optimizer
    model = util.select_model(args, rank)
    if args.model_init_path is not None:
        model.load_state_dict(torch.load(args.model_init_path, map_location='cpu'))
        logging.info("Loaded model from {}".format(args.model_init_path))
    
    if args.gpu:
        model = model.cuda()    
    
    if hasattr(dataset, 'criterion'):
        criterion = dataset.criterion()
    else:
        criterion = nn.CrossEntropyLoss().cuda() if(args.gpu) else nn.CrossEntropyLoss()

    # select optimizer according to algorithm
    algorithms = {
        'fedavg': FedAvg, 
        'fedprox': FedProx,
        'fednova': FedNova,
        'fedadam' : FedAdam,
        'fedyogi' : FedYogi,
        'localSGD' : LocalSGD,
        'scaffold' : Scaffold
    }
    selected_opt = algorithms[args.optimizer]
    optimizer = selected_opt(model.parameters(),
                             lr=args.lr,
                             gmf=args.gmf,
                             mu=args.mu,
                             ratio=None, # Set after sampling
                             momentum=args.momentum,
                             nesterov = False,
                             weight_decay=1e-4,
                             slr=args.slr,
                             clients_per_round=args.numclients,
                             total_clients=args.totalclients,)

    if args.use_scheduler and args.optimizer == 'localSGD':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.highE, eta_min=0.001)
    else:
        scheduler = None

    # setup wandb metrics
    wandb.define_metric('train_step')
    wandb.define_metric('train_acc', summary='max', step_metric='train_step')
    wandb.define_metric('train_loss', step_metric='train_step')
    
    wandb.define_metric('val_step')
    wandb.define_metric('val_acc', summary='max', step_metric='val_step')
    
    wandb.define_metric('test_step')
    if args.rounds == 1 or rank == 0:
        wandb.define_metric('test_acc', summary='max', step_metric='test_step')
    
    if args.rounds > 1 and rank == 0:
        wandb.define_metric('global_val_acc', summary='max', step_metric='test_step')

    # start training
    best_test_accuracy = 0; args.global_epoch = 0; args.global_itr = 0
    for rnd in range(args.rounds):
        
        if args.optimizer == 'scaffold' or args.rounds == 1: # local training or scaffold
            selected_client_indexes = np.arange(total_clients) # scaffold must fix process --> client mapping
        else:
            selected_client_indexes = rng.choice(range(total_clients), size=args.numclients, replace=False)
    
        # This process trains only the client corresponding to its rank
        client_index = selected_client_indexes[rank]
        train_loader, proxy_loader, val_loader, test_loader, local_test_loader, num_samples = \
            dataset.fetch(client_index)
        num_samples_this_round = num_samples[selected_client_indexes]
        logging.debug(f'Number of samples this round: {num_samples_this_round}')

        # Decide number of local updates per client
        local_epochs = get_local_epochs(rng)

        # evaluate and set the ratio needed for aggregating
        ratios, tau_i = get_ratios(local_epochs, num_samples_this_round, rank, args.max_itr)
        optimizer.set_ratio(ratios[rank])
        logging.debug("Client id {} local iterations {} weight {:.4f}"
                      .format(client_index, tau_i, ratios[rank]))

        # Decay learning rate according to round index
        if args.updatelr:
            update_learning_rate(optimizer, rnd, args.lr)
        
        # Clients locally train for several local epochs
        best_model = train(model, criterion, optimizer, train_loader, tau_i, \
            client_index, local_test_loader, metric=metric, scheduler=scheduler,
            require_argmax=require_argmax, is_nlp=is_nlp)
        
        # synchronize parameters
        maybe_barrier()
        comm_start = time.time()
        optimizer.average()
        maybe_barrier()
        comm_end = time.time()
        comm_time = comm_end - comm_start
        
        # evaluate test accuracy
        if rnd % args.evalafter == 0 or rnd == args.rounds - 1:
            # evaluate for rank 0 when doing FL i.e. args.rounds > 1
            # or evaluate for all ranks when doing local training i.e. args.rounds == 1
            if rank == 0 or args.rounds == 1:
                eval_model = best_model if args.rounds == 1 and best_model else model # use best model when local training
                eval_model = eval_model.cuda() if args.gpu else eval_model
                test_acc = evaluate(eval_model, test_loader, metric=metric, \
                                    require_argmax=require_argmax, is_nlp=is_nlp)
                if test_acc > best_test_accuracy:
                    best_test_accuracy = test_acc
            
                # record metrics
                logging.info("Round {} test accuracy {:.3f} time {:.3f}".format(rnd, test_acc, comm_time))
                with open(args.out_fname_test, '+a') as f:
                    print('{ep},{val:.4f},{time:.4f}'
                        .format(ep=rnd,
                            val=test_acc, time=comm_time), file=f)
                
                # global validation accuracy (only for FL)
                # for local training, we report local validation accuracy evaluated at the end of training
                if val_loader and args.rounds > 1:
                    val_acc = evaluate(model, val_loader, metric=metric, \
                                       require_argmax=require_argmax, is_nlp=is_nlp)
                    with open(args.out_fname_global_val, '+a') as f:
                        print('{ep},{test:.4f},{time:.4f}'
                            .format(ep=rnd,
                                test=val_acc, time=comm_time), file=f)
                    wandb.log({'test_acc': test_acc, 'global_val_acc': val_acc, 'test_step': rnd+1})
                else:
                    wandb.log({'test_acc': test_acc, 'test_step': rnd+1})
                                                
        logging.info("Rnd {} Worker {} Client id {} best test accuracy {:.3f}"
                 .format(rnd, rank, client_index, best_test_accuracy))

        if rank==0 and args.stop_early:
            if(check_stopping_criteria(args.optimizer, args.dataset, args.alpha, best_test_accuracy, rnd)):
                sys.exit('Accuracy not improving. Stopping early.')

    # Save the model for fine-tuning (for FL algorithms)
    if args.save_model and args.rounds != 1 and rank == 0:
        torch.save(model.state_dict(), args.out_fname_model)
        logging.info("Saved model {} to {}".format(rank, args.out_fname_model))

    # Save the model after training (only for local training based methods)
    if args.save_model and args.rounds == 1 and args.optimizer == 'localSGD':
        # save final model
        torch.save(model.state_dict(), args.out_fname_model)
        logging.info("Saved model {} to {}".format(rank, args.out_fname_model))

        # save best model
        if best_model:
            torch.save(best_model.state_dict(), f'{folder_name}/{rank}_best_model.pth')
            logging.info("Saved best model {} to {}".format(rank, f'{folder_name}/{rank}_best_model.pth'))

        # save the label distribution as a list
        assert client_index == rank, "Client index and rank should be the same for local training"
        label_dist = dataset.get_label_dist(client_index)
        with open(args.out_fname_label_dist, 'wb') as f:
            pickle.dump(label_dist, f)

    logging.info("Awaiting final synchronisation barrier.")
    maybe_barrier()

def evaluate(model, test_loader, metric=None, require_argmax=True, is_nlp=False):
    model.eval()

    y_preds = []
    y_trues = []
    with torch.no_grad():
        for d in test_loader:
            if is_nlp:
                ids, mask, target = d['ids'], d['mask'], d['targets']
            else:
                data, target = d
            if(args.gpu):
                if is_nlp:
                    ids = ids.cuda(non_blocking = True)
                    mask = mask.cuda(non_blocking = True)
                elif isinstance(data, list):
                    data = [d.cuda(non_blocking = True) for d in data]
                else:
                    data = data.cuda(non_blocking = True)
            
            if is_nlp:
                outputs = model(ids, mask)
            else:
                outputs = model(data)

            y_trues.append(target)
            if require_argmax:
                y_preds.append(torch.argmax(outputs, dim=1).cpu())
            else:
                y_preds.append(outputs.cpu())

    y_preds_np = np.concatenate(y_preds)
    if len(y_preds_np.shape) == 2 and y_preds_np.shape[1] == 1:
        y_preds_np = np.squeeze(y_preds_np, axis=1)

    y_trues_np = np.concatenate(y_trues)
    if len(y_trues_np.shape) == 2 and y_trues_np.shape[1] == 1:
        y_trues_np = np.squeeze(y_trues_np, axis=1)
    
    if metric is not None:
        acc1 = metric(y_trues_np, y_preds_np)
    else:
        acc1 = accuracy_score(y_trues_np, y_preds_np)
        
    return acc1*100

def train(model, criterion, optimizer, loader, iterations, client_id,
          local_test_loader=None, metric=None, scheduler=None, require_argmax=True, 
          is_nlp=False):

    model.train()
    # allocating continuous space in RNN models
    if(hasattr(model, 'lstm')): 
        model.lstm.flatten_parameters() # no gpu check needed since it is a no-op when cpu

    best_model = None
    best_val_acc = 0.0
    val_acc = 0.0
    losses = util.Meter(ptag='Loss')
    top1 = util.Meter(ptag='Prec@1')
    count = 0
    epoch = -1

    while count < iterations:
        epoch += 1; args.global_epoch += 1
        model.train() # evaluate calls model.eval()
        
        for batch_idx, d in enumerate(loader):
            if is_nlp:
                ids, mask, target = d['ids'], d['mask'], d['targets']
            else:
                data, target = d
            count += 1; args.global_itr += 1

            # data loading for GPU
            if(args.gpu):
                if is_nlp:
                    ids = ids.cuda(non_blocking = True)
                    mask = mask.cuda(non_blocking = True)
                elif isinstance(data, list):
                    data = [d.cuda(non_blocking = True) for d in data]
                else:
                    data = data.cuda(non_blocking = True)
                target = target.cuda(non_blocking = True)

            # forward pass
            if is_nlp:
                output = model(ids, mask)
            else:
                output = model(data)
            loss = criterion(output, target)

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()

            # write log files
            train_acc = util.comp_accuracy(output, target)[0].item()

            losses.update(loss.item(), target.size(0))
            top1.update(train_acc, target.size(0))

            if batch_idx % args.print_freq == 0 and args.save:
                logging.debug('epoch {} itr {}, '
                            'rank {}, loss value {:.4f}, train accuracy {:.3f}'
                            .format(epoch, batch_idx, rank, losses.avg, top1.avg))

                with open(args.out_fname, '+a') as f:
                    print('{ep},{itr},'
                        '{loss.val:.4f},{loss.avg:.4f},'
                        '{top1.val:.3f},{top1.avg:.3f},{cid}'
                        .format(ep=epoch, itr=batch_idx,
                                loss=losses, top1=top1, cid=client_id), file=f)
                
                if not args.isEpochs: 
                    wandb.log({'train_acc': top1.avg, 'train_loss': losses.avg, 'train_step': args.global_itr})    

            if count >= iterations:
                break
        
        if scheduler:
            scheduler.step()
            logging.debug(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

        # evaluate on local test set
        if local_test_loader: # name of variable to be changed
            logging.debug("Evaluating on local validation set")
            val_acc = evaluate(model, local_test_loader, metric=metric, require_argmax=require_argmax, is_nlp=is_nlp)
            
            with open(args.out_fname_val, '+a') as f:
                print('{ep},{val:.4f}'.format(ep=epoch,val=val_acc), file=f)
            
            logging.info("epoch {} client id {} local validation accuracy {:.3f}"
                    .format(epoch, client_id, val_acc))
            
            if not args.isEpochs: wandb.log({'val_acc': val_acc, 'val_step': args.global_itr})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model).cpu()
                logging.info("Best model updated !")
        
        if args.isEpochs:
            wandb.log({'train_acc': top1.avg, 'train_loss': losses.avg, 'train_step': args.global_epoch})
            wandb.log({'val_acc': val_acc, 'val_step': args.global_epoch})

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},'
            '{loss.val:.4f},{loss.avg:.4f},'
            '{top1.val:.3f},{top1.avg:.3f},{cid}'
            .format(ep=epoch, itr=batch_idx,
                    loss=losses, top1=top1, cid=client_id), file=f)
    
    if not args.isEpochs: 
        wandb.log({'train_acc': top1.avg, 'train_loss': losses.avg, 'train_step': args.global_itr})

    return best_model if best_model is not None else model

def get_local_epochs(rng):
        return rng.integers(low=args.lowE, high=args.highE+1, size=args.numclients)

def get_ratios(local_epochs, num_samples_this_round, rank, tau_max=0):
    assert(len(local_epochs) == len(num_samples_this_round))

    num_seen_samples = []; tau_i = -1
    if(args.isEpochs):
        for i, epochs_i in enumerate(local_epochs):
            n_i = num_samples_this_round[i]
            tau_possible_i = ceil(n_i / args.bs) 
            tau_required_i = max(floor(epochs_i * tau_possible_i), 1) # has to be atleast 1
            
            if(tau_max and tau_required_i > tau_max): # cap with max when specified
                tau_required_i = tau_max

            if(tau_possible_i <= tau_required_i): # more than or equal to one epoch
                num_seen_samples.append(n_i)
            else:
                num_seen_samples.append(args.bs * max(tau_required_i,1)) # should see at least one batch
            
            if(i == rank): tau_i = tau_required_i 
    else:
        for i, n_i in enumerate(num_samples_this_round):
            tau_possible_i = ceil(n_i / args.bs)
            tau_required_i = max(floor(local_epochs[i]), 1) # has to be atleast 1
            if(tau_possible_i <= tau_required_i): # more than or equal to one epoch
                num_seen_samples.append(n_i)
            else:
                num_seen_samples.append(args.bs * max(tau_required_i, 1)) # should see at least one batch

            if(i == rank): tau_i = tau_required_i
    
    num_seen_samples = np.array(num_seen_samples)
    ratios = num_seen_samples / np.sum(num_seen_samples)

    if(args.weights == 'uniform'):
        ratios = np.array([1.0/args.numclients for _ in range(args.numclients)])

    return ratios, tau_i

def update_learning_rate(optimizer, epoch, target_lr):
    """
    1) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: target_lr is the reference learning rate from which to scale down
    """
    if epoch == int(args.rounds / 2):
        lr = target_lr/10
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == int(args.rounds * 0.75):
        lr = target_lr/100
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    # LocalSGD does no cross-client communication (optimizer.average() is a no-op) --
    # forming a torch.distributed group just to synchronize barriers that guard no data
    # exchange forces all `size` clients to be simultaneously resident in memory for no
    # functional reason, so skip it and let each client run fully independently.
    if args.optimizer != 'localSGD':
        # large timeout required for alpha = 0.01
        dist.init_process_group(backend=args.backend,
                                init_method=args.initmethod,
                                rank=rank,
                                world_size=size,
                                timeout=datetime.timedelta(seconds=3600*8)) # 8 hours
    fn(rank, size)

if __name__ == "__main__":
    rank = args.rank
    size = args.size
    print(rank)
    
    if(args.gpu): # gpu
        if(not torch.cuda.is_available()):
            raise ValueError(f"args.gpu set to {args.gpu}. GPU not available/detected on this machine.")
        else:
            gpu_count = torch.cuda.device_count()
            torch.cuda.set_device(rank % gpu_count)
    
    else: # cpu
        total_threads = os.cpu_count()
        if args.procs_per_machine == -1: # set 1 thread per process
            threads_per_proc = 1
        else:
            threads_per_proc = max(floor(total_threads/args.procs_per_machine), 1)
        torch.set_num_threads(threads_per_proc)
        torch.set_num_interop_threads(1)
        logging.info(f"Assigned {threads_per_proc} threads per process.")
    
    # initialize wandb
    wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity,
        group=args.name,
        name=f"process_r{rank}",
        job_type="train",
        config=args,
        mode="disabled" if args.disable_wandb else "online"
    )

    init_processes(rank, size, run)

    wandb.finish()