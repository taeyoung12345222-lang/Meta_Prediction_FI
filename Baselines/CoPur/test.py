import wandb
import torch
import numpy as np
import os, sys
import logging
from sklearn.metrics import accuracy_score

from args import get_copur_args
from Models.server_model import CoPurAggregator
from Models.autoencoder import CoPurAE
from Defenses.copur import purify
from Defenses.manifold_projection import manifold_projection
from Attacks.dfl import distributed_feature_flipping
from Attacks.sia import sia_attack, sia_attack_blackbox

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Utils.general import get_num_classes

def evaluate_avg(test_loader, num_clients, num_classes):
    y_preds = []
    y_trues = []

    for (data, target) in test_loader:
        data = data.reshape(-1, num_clients, num_classes).numpy()
        avg_data = data.mean(axis=1)
        y_preds.append(torch.argmax(torch.tensor(avg_data), dim=1).cpu())
        y_trues.append(target.cpu())

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)

    acc1 = accuracy_score(y_trues_np, y_preds_np)
    
    return acc1

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

    testset = torch.load(os.path.join(args.datapath, 'logit_testset.pth'), map_location='cpu')

    # evaluate accuracy of averaging
    averaged_logits = [(x.reshape(-1, n_classes).mean(dim=0).squeeze(), y) for x, y in testset] 
    y_preds = [torch.argmax(x) for x, _ in averaged_logits]
    y_trues = [y for _, y in averaged_logits]
    acc_avg = accuracy_score(y_trues, y_preds)
    logging.info('Averaged logits accuracy: {:.3f}'.format(acc_avg))

    shuffle_order = rng.permutation(n_clients)
    logging.info(f'Shuffle order: {shuffle_order}')

    in_dim = n_classes * n_clients
    out_dim = n_classes
    
    ae_model = CoPurAE(in_dim, in_dim, args.hidden_dim, args.encode_dim)
    ae_model.load_state_dict(torch.load(os.path.join(args.modelpath, 'model.pth'), map_location='cpu'))
    
    model = CoPurAggregator(in_dim, out_dim)
    model.load_state_dict(torch.load(os.path.join(args.modelpath2, 'model.pth'), map_location='cpu'))

    device = 'cuda' if args.gpu else 'cpu'

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    acc_mp = manifold_projection(test_loader, model, ae_model, device, n_clients, n_classes)
    logging.info('After Manifold Projection, Acc on original testset: {:.3f}'.format(acc_mp))

    acc_copur = purify(test_loader, model, ae_model, args.lr, torch.nn.MSELoss(), device,
                 args.initial_iters, args.final_iters, n_clients, n_classes, args.tau)
    logging.info('Accuracy after CoPur on original testset: {:.3f}'.format(acc_copur))
    
    results = {0: (acc_avg, acc_mp, acc_copur)}
    
    if args.eval_one_adv and args.n_adv >= 1:
        ns_adv = [args.n_adv]
    else:
        ns_adv = list(range(1, args.n_adv + 1)) ### [1,N_ADV]
    
    for n_adv in ns_adv:
        logging.info(f'==> Working on {n_adv} Byzantine clients')
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)    
        
        if args.attack_type == 'dfl':
            test_loader = distributed_feature_flipping(test_loader, args.amplification, n_adv, n_clients, n_classes, shuffle_order)
            
            logging.info('='*10 + '# N_adv = {}'.format(n_adv))

            acc_avg = evaluate_avg(test_loader, n_clients, n_classes)
            logging.info('After DFL attack, Averaged logits accuracy: {:.3f}'.format(acc_avg))
            
            criterion = torch.nn.MSELoss()

            acc_mp = manifold_projection(test_loader, model, ae_model, device, n_clients, n_classes)
            logging.info('Accuracy after Manifold Projection, Acc: {:.3f}'.format(acc_mp))

            acc_copur = purify(test_loader, model, ae_model, args.lr, criterion, device, 
                         args.initial_iters, args.final_iters, n_clients, n_classes, args.tau)
            logging.info('Accuracy after CoPur: {:.3f}'.format(acc_copur))

            results[n_adv] = (acc_avg, acc_mp, acc_copur)
        
        elif args.attack_type == 'sia':
            if args.black_box:
                logging.info("Using black-box SIA attack")
                test_loader = sia_attack_blackbox(test_loader, n_adv, n_clients, n_classes, shuffle_order, args.amplification)
            else:
                logging.info("Using white-box SIA attack")
                test_loader = sia_attack(test_loader, n_adv, n_clients, n_classes, 
                                    shuffle_order, model, ae_model, args.lr, torch.nn.MSELoss(),
                                    device, args.initial_iters, args.final_iters, args.tau, args.amplification)
            
            logging.info('='*10 + '# N_adv = {}'.format(n_adv))

            acc_avg = evaluate_avg(test_loader, n_clients, n_classes)
            logging.info('After SIA attack, Averaged logits accuracy: {:.3f}'.format(acc_avg))
            
            criterion = torch.nn.MSELoss()

            acc_mp = manifold_projection(test_loader, model, ae_model, device, n_clients, n_classes)
            logging.info('Accuracy after Manifold Projection, Acc: {:.3f}'.format(acc_mp))

            acc_copur = purify(test_loader, model, ae_model, args.lr, criterion, device, 
                         args.initial_iters, args.final_iters, n_clients, n_classes, args.tau)
            logging.info('Accuracy after CoPur: {:.3f}'.format(acc_copur))

            results[n_adv] = (acc_avg, acc_mp, acc_copur)

    print(results)

    # save to a csv file
    with open(os.path.join(args.save_dir, 'results.csv'), 'w') as f:
        f.write('n_adv,acc_avg,acc_mp,acc_copur\n')
        for n_adv in results:
            acc_avg, acc_mp, acc_copur = results[n_adv]
            f.write(f'{n_adv},{acc_avg:.3f},{acc_mp:.3f},{acc_copur:.3f}\n')

    if args.wandb: wandb.finish()