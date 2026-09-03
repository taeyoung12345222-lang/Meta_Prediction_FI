import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import torch.nn.functional as F

from Utils.fl_args import get_args
from Utils.adversarial import adversarial_attack, class_prior_attack, compute_similarity_matrix, impersonation_attack, loss_maximization_attack, sia_attack, Carlini_Wagner_loss, \
    sia_attack_blackbox, sia_attack_blackbox_collude, cross_entropy_from_probs
from Utils.general import get_model, get_num_classes
from aggregator_training_fl import _evaluate, generate_subsets_with_masks
from Baselines.CoPur.Attacks.dfl import distributed_feature_flipping_with_data_loading

if __name__ == "__main__":
    
    args = get_args()

    # save all params to params.txt
    with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
        str_args = '\n'.join([f'{k}: {v}' for k, v in vars(args).items()])
        f.write(str_args)

    # make training deterministic
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_classes = get_num_classes(args.dataset)
    n_clients = args.size
    rng = np.random.default_rng(seed=args.seed)
    
    output_prob = args.model in ['F_TM', 'F_Median', 'F_Median2'] and args.normalize and args.normalization_type == 'simplex'
    f = get_model(args.model, n_clients, n_classes, args.trim_ratio, args.dim_hidden, args.lip_scale, args.n_adv, output_prob)

    if args.loss_fn == 'cw': ## attack trainable models using Carlini-Wagner loss
        print('==> Using Carlini-Wagner loss')
        if args.model in ['F_Avg', 'F_TM', 'F_Median'] and args.normalize and args.normalization_type == 'simplex':
            adv_criterion = lambda logits, labels: Carlini_Wagner_loss(logits, labels, input_is_prob=True, confidence=args.cw_confidence)
        else: # for DeepSet
            adv_criterion = lambda logits, labels: Carlini_Wagner_loss(logits, labels, input_is_prob=False, confidence=args.cw_confidence)
    else:
        print('==> Using cross-entropy loss')
        if args.model in ['F_Avg', 'F_TM', 'F_Median'] and args.normalize and args.normalization_type == 'simplex':
            adv_criterion = cross_entropy_from_probs
        else:
            adv_criterion = F.cross_entropy 

    device = 'cuda' if args.gpu else 'cpu'
    
    # configure logging
    logging.basicConfig(
        filename='{}/log2.txt'.format(args.save_dir),
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
    if args.normalize:
        logging.info('==> Normalizing trainset')
        if args.normalization_type in ['simplex']:
            trainset = [(torch.softmax(x, dim=-1), y) for x, y in trainset]
        elif args.normalization_type in ['simplex-one-hot']:
            trainset = [(torch.softmax(x, dim=-1), y) for x, y in trainset]
            trainset = [(F.one_hot(torch.argmax(x, dim=-1), num_classes=n_classes).float(), y) for x, y in trainset]
        else: raise ValueError('Normalization type not supported')

        # check if normalization is correct
        for i in range(10): 
            logging.debug('Norm: {} \n Max: {} \n Min: {} \n Sum: {} \n \n -------- \n'.format(
                torch.norm(trainset[i][0], p=2, dim=-1).tolist()[:5], 
                torch.max(trainset[i][0], dim=-1).values.tolist()[:5], 
                torch.min(trainset[i][0], dim=-1).values.tolist()[:5],
                torch.sum(trainset[i][0], dim=-1).tolist()[:5]
            ))
    trainset = generate_subsets_with_masks(trainset, rng, n_clients, n_clients, n_clients, 1) # just includes masks
    logging.info(f'Trainset size: {len(trainset)}')

    testset = torch.load(os.path.join(args.datapath, 'logit_testset.pth'), map_location='cpu')
    testset = [(x.reshape(-1, n_classes), y) for x, y in testset]
    if args.normalize:
        logging.info('==> Normalizing testset')
        if args.normalization_type in ['simplex']:
            testset = [(torch.softmax(x, dim=-1), y) for x, y in testset]
        elif args.normalization_type in ['simplex-one-hot']:
            testset = [(torch.softmax(x, dim=-1), y) for x, y in testset]
            testset = [(F.one_hot(torch.argmax(x, dim=-1), num_classes=n_classes).float(), y) for x, y in testset]
        # check if normalization is correct
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
    print('Averaged logits accuracy: {:.3f}'.format(acc))

    # shuffle the rows of each x
    shuffle_order = rng.permutation(testset[0][0].shape[0])
    logging.info(f'Shuffle order: {shuffle_order}')
    shuffled_testset = [(x[shuffle_order], mask[shuffle_order], y) for x, mask, y in testset]

    # create data loaders
    test_loader = torch.utils.data.DataLoader(shuffled_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # load model
    if f.state_dict() != {}: # not a static aggregation
        print(f'==> Not a static aggregation, loading model for {args.model}')
        f.load_state_dict(torch.load(os.path.join(args.modelpath, 'model.pth'), map_location='cpu'))
    else:
        print(f'==> Static aggregation, no model to load for {args.model}')
    f = f.to(device)

    acc = _evaluate(f, test_loader, device)
    logging.info(f'Initial accuracy: {acc:.3f}')
    print(f'Accuracy with no adversaries: {acc:.3f}')
    results = {0: acc}

    # performance eval after dropping clients
    if args.eval_one_adv and args.n_adv >= 1:
        ns_adv = [args.n_adv]
    else:
        ns_adv = list(range(1, args.n_adv + 1)) ### [1,N_ADV]
    
    for n_adv in ns_adv:
        logging.info(f'==> Working on {n_adv} dropped clients')
        
        ## ------------ in-place or new ------------
        if args.attack_type == 'pgd': # white-box attack
            if args.new_adversaries: # add new adversaries i.e. adversaries are different from honest clients
                dropped_testset = shuffled_testset
            else: # honest clients are dropped and replaced with adversaries
                dropped_testset = [(x[:-n_adv], mask[:-n_adv], y) for x, mask, y in shuffled_testset]
        elif args.attack_type in ['sia', 'dfl', 'lma', 'cpa', 'ia']: # adversaries are always in-place
            dropped_testset = shuffled_testset

        # create data loaders
        test_loader = torch.utils.data.DataLoader(dropped_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        acc = _evaluate(f, test_loader, device)
        logging.info(f'Accuracy after dropping {n_adv} clients: {acc:.3f}')

        # launch adversarial attack
        if args.adversarial:
            if args.attack_type == 'pgd':
                ## ------------ black-box vs white-box ------------
                if args.black_box:
                    raise ValueError("PGD attack can only be launched as a white-box attack")

                logging.info('==> Generating adversarial examples')
                test_loader = adversarial_attack(f, test_loader, device, n_adv, adv_criterion, n_iter=args.n_iter, 
                                                alpha=0.1, # alpha here is the step size
                                                eps=0.1)
                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')

            elif args.attack_type == 'sia':
                logging.info('==> Attacking using SIA')
                ## ------------ black-box vs white-box ------------
                if args.black_box and not args.collude:
                    test_loader = sia_attack_blackbox(test_loader, device, n_adv)
                elif args.black_box and args.collude:
                    test_loader = sia_attack_blackbox_collude(f, test_loader, device, n_adv)
                else: # white-box (always collude)
                    test_loader = sia_attack(f, test_loader, device, n_adv)
                
                if args.save_attacked_logits:
                    logging.info('==> Saving attacked logits and labels as list of tuples')
                    attacked_data = []
                    for x, _, y in test_loader:
                        for xi, yi in zip(x, y):
                            attacked_data.append((xi, yi))
                    
                    save_path = os.path.join(args.save_dir, f'nadv{n_adv}_M{n_clients}_attacked_logits.pth')
                    torch.save(attacked_data, save_path)

                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
            
            elif args.attack_type == 'dfl':
                logging.info('==> Atacking using Distributed Feature Flipping (DFL)')
                adv_testset = distributed_feature_flipping_with_data_loading(
                    args.datapath, shuffle_order, # this attack must be done in the logit space
                    args.amplification, n_adv, n_clients, n_classes)
                adv_testset = [(torch.softmax(x, dim=-1), y) for x, y in adv_testset]
                adv_testset = generate_subsets_with_masks(adv_testset, rng, n_clients, n_clients, n_clients, 1) # just includes masks
                test_loader = torch.utils.data.DataLoader(adv_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')

            elif args.attack_type == 'lma':
                ## ------------ black-box vs white-box ------------
                if args.black_box:
                    raise ValueError("LMA can only be launched as a white-box attack")
                
                logging.info('==> Attacking using Loss Maximization Attack (LMA)')
                test_loader = loss_maximization_attack(f, test_loader, device, n_adv)

                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')

            elif args.attack_type == 'cpa':
                ## ------------ black-box vs white-box ------------
                if args.black_box:
                    raise ValueError("CPA can only be launched as a white-box attack")
                
                logging.info('==> Attacking using Class Prior Attack (CPA)')
                # Load similarity matrix
                S = torch.load(args.S_path, map_location='cpu')
                S = S.to(device)
                test_loader = class_prior_attack(f, test_loader, device, n_adv, S)

                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
            
            elif args.attack_type == 'ia':
                ## ------------ black-box vs white-box ------------
                if args.black_box:
                    raise ValueError("IA can only be launched as a white-box attack")
                
                logging.info('==> Attacking using Impersonation Attack (IA)')
                test_loader = impersonation_attack(None, test_loader, device, n_adv)

                acc = _evaluate(f, test_loader, device)
                logging.info(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                print(f'Accuracy with {n_adv} adversarial clients: {acc:.3f}')
                

        results[n_adv] = acc
    
    # save as csv
    df = pd.DataFrame(list(results.items()), columns=['n_dropped', 'accuracy'])
    df.to_csv(os.path.join(args.save_dir, 'results.csv'), index=False)


