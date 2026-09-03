import util_v4 as util
import logging

from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import copy

################# HELPER FUNCTIONS #################

# faster version that uses precomputed logits
def evaluate_competencies_v2(client_index, num_classes, total_clients, trainset):
    total_correct = 0
    total_predicted = 0
    
    competency_matrix = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_samples_per_class = [0.0 for _ in range(num_classes)]

    for X, y in trainset:
        X = torch.reshape(X, (-1, total_clients, num_classes))
        # get prediction for client
        X = X[:, client_index, :]
        # get predicted label
        y_pred = torch.argmax(X, dim=1)

        for ground_truth, prediction in zip(y, y_pred):
            prediction = int(prediction.item())
            ground_truth = int(ground_truth.item())
            competency_matrix[ground_truth][prediction] += 1.0
            total_samples_per_class[ground_truth] += 1.0
            if ground_truth == prediction:
                total_correct += 1
            total_predicted += 1
    
    seen_classes = [k for k in range(num_classes) if total_samples_per_class[k] != 0]
    unseen_classes = [k for k in range(num_classes) if total_samples_per_class[k] == 0]
    logging.debug(f"Total seen classes {len(seen_classes)} and unseen classes {len(unseen_classes)}")

    for k in seen_classes:
        for j in seen_classes:
            competency_matrix[k][j] /= total_samples_per_class[k]
    
    for k in unseen_classes:
        for j in seen_classes:
            competency_matrix[k][j] = 1.0/len(seen_classes)
    
    accuracy = (total_correct/total_predicted)*100
    logging.debug("Competence Accuracy: {:.2f}".format(accuracy))
    return competency_matrix

# Used in polychotomous voting aggregation
def evaluate_competencies(model, dataset, num_classes, use_gpu=False, device=None, is_binary_classification=False):

    total_correct = 0
    total_predicted = 0

    if use_gpu and device is None:
        device = torch.device('cpu')

    competency_matrix = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_samples_per_class = [0.0 for _ in range(num_classes)]

    model = model.to(device)
    with torch.no_grad():
        for data, target in dataset:
            if use_gpu:
                data = data.to(device)
                target = target.to(device)
            output = model(data).detach().cpu()

            if not is_binary_classification:        
                _, predictions = torch.max(output, 1)
            else: # binary classification
                predictions = torch.round(torch.sigmoid(output))

            for ground_truth, prediction in zip(target, predictions):
                prediction = int(prediction.item())
                ground_truth = int(ground_truth.item())
                competency_matrix[ground_truth][prediction] += 1.0
                total_samples_per_class[ground_truth] += 1.0
                if ground_truth == prediction:
                    total_correct += 1
                total_predicted += 1
    
    seen_classes = [k for k in range(num_classes) if total_samples_per_class[k] != 0]
    unseen_classes = [k for k in range(num_classes) if total_samples_per_class[k] == 0]
    logging.debug(f"Total seen classes {len(seen_classes)} and unseen classes {len(unseen_classes)}")
    
    for k in seen_classes:
        for j in seen_classes:
            competency_matrix[k][j] /= total_samples_per_class[k]

    for k in unseen_classes:
        for j in seen_classes:
            competency_matrix[k][j] = 1.0/len(seen_classes)

    accuracy = (total_correct/total_predicted)*100
    logging.debug("Competence Accuracy: {:.2f}".format(accuracy))
    return competency_matrix

def get_competencies(classes, competency_matrix):
    num_classes = len(competency_matrix)
    competencies_to_send = [[competency_matrix[i][k] for i in range(num_classes)] for k in classes]
    return competencies_to_send

def get_prediction_using_competency(competencies, num_classes, return_confidence=False):
    classwise_estimates = []
    for m in range(num_classes):
        ans = 1.0
        for node_competency in competencies:
            ans *= node_competency[m]    
        classwise_estimates.append(ans)
    
    if return_confidence:
        confidences = torch.softmax(torch.tensor(classwise_estimates), dim=0)
        return torch.argmax(confidences).item(), confidences
    else:
        return np.argmax(classwise_estimates)

def _one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# Used in linear mapping aggregation
def _check_accuracy(weights, bias, testset, total_clients, num_classes, metric=None, require_argmax=True):
        softmax = torch.nn.Softmax(dim=1)
        y_preds = []
        y_trues = []

        with torch.no_grad():
            for X, y in testset:
                X = torch.reshape(X, (-1, total_clients, num_classes))
                X = torch.swapaxes(X, 1, 2)
                X = X*weights
                y_pred_probits = torch.sum(X, dim=2) + bias
                y_pred_probits = softmax(y_pred_probits)
                y_pred = torch.argmax(y_pred_probits, dim=1) if require_argmax else y_pred_probits
                y_preds.append(y_pred)
                y_trues.append(y)

        y_preds_np = np.concatenate(y_preds)
        y_trues_np = np.concatenate(y_trues)
        if metric is not None:
            linear_mapping_performance = metric(y_trues_np, y_preds_np)
        else:
            linear_mapping_performance = accuracy_score(y_trues_np, y_preds_np)
        return linear_mapping_performance

# Used in NN aggregation
def _train_and_evaluate(model, criterion, optimizer, scheduler, 
                        train_loader, test_loader, iterations, device, test_every=5, metric=None, require_argmax=True):

    model.train()
    model = model.to(device)
    best_model = None

    losses = util.Meter(ptag='Loss')
    top1 = util.Meter(ptag='Prec@1')
    epoch = 0

    best_acc = 0.0

    while epoch < iterations:
        epoch += 1
        for batch_idx, (data, target) in enumerate(train_loader):
            # data loading for GPU
            data = data.to(device)
            target = target.to(device)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()

            # write log files
            train_acc = util.comp_accuracy(output, target)
            
            losses.update(loss.item(), data.size(0))
            top1.update(train_acc[0].item(), data.size(0))

        scheduler.step()

        logging.debug('Epoch {ep} Loss {loss.avg:.4f} Train Acc {top1.avg:.3f}'.format(
            ep=epoch, loss=losses, top1=top1))

        if epoch % test_every == 0:
            test_acc = _evaluate(model, test_loader, device, metric=metric, require_argmax=require_argmax)
            if test_acc > best_acc:
                best_acc = test_acc
                # store the best model by making a copy
                best_model = copy.deepcopy(model)
            logging.info('Epoch {ep} Test Acc {acc:.3f}'.format(ep=epoch, acc=test_acc))

    return best_acc, best_model

def _evaluate(model, test_loader, device, metric=None, require_argmax=True):
    model.eval()
    model.to(device)

    y_preds = []
    y_trues = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            outputs = model(data)

            y_trues.append(target)
            
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

################# AGGREGATIONS #################

def evaluate_all_aggregations(val_loader, test_loader, models, label_dists, args, \
                              training_params=None, device=None, metric=None, 
                              require_argmax=True, lossfunc=None):

    if device is None:
        device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    models = [model.to(device) for model in models]
    
    # Create the dataset of predictions for training and testing
    logging.info("Computing trainset logits")
    trainset = []
    with torch.no_grad():
        for elems, labels in val_loader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            trainset.append((stacked_outputs, labels))

    logging.info("Computing testset logits")
    testset = []
    with torch.no_grad():
        for elems, labels in test_loader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            testset.append((stacked_outputs, labels))

    results = {}
    num_classes = util.get_num_classes(args.dataset)
    total_clients = args.totalclients
    #----------------------------------------------------------------------------
    ## Averaging
    logging.info("Evaluating Averaging strategy")
    y_preds = []
    y_trues = []
    
    for X, y in testset:
        X = torch.reshape(X, (-1, total_clients, num_classes))
        y_avg = torch.mean(X, dim=1)
        y_pred = torch.argmax(y_avg, dim=1) if require_argmax else y_avg
        y_preds.append(y_pred)
        y_trues.append(y)

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)
    if metric is not None:
        avg_performance = metric(y_trues_np, y_preds_np)
    else:
        avg_performance = accuracy_score(y_trues_np, y_preds_np)
    print(f'==> Averaging Performance: {avg_performance}')
    results['averaging'] = avg_performance

    #----------------------------------------------------------------------------
    ## Weighted Averaging
    logging.info("Evaluating Weighted Averaging strategy")

    # Get weights from labels
    my_labels_tensor = torch.tensor(label_dists)
    label_sum_tensor = my_labels_tensor.sum(dim=0)
    my_weights_tensor = my_labels_tensor / label_sum_tensor

    y_preds = []
    y_trues = []
    for X, y in testset:
        X = torch.reshape(X, (-1, total_clients, num_classes))
        X = X*my_weights_tensor
        y_avg = torch.mean(X, dim=1)
        y_pred = torch.argmax(y_avg, dim=1) if require_argmax else y_avg
        y_preds.append(y_pred)
        y_trues.append(y)

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)
    if metric is not None:
        avg_performance = metric(y_trues_np, y_preds_np)
    else:
        avg_performance = accuracy_score(y_trues_np, y_preds_np)
    print(f'==> Weighted Averaging Performance: {avg_performance}')
    results['weighted_averaging'] = avg_performance

    #----------------------------------------------------------------------------
    # Polychotomous Voting
    logging.info("Evaluating Polychotomous Voting strategy")

    competency_matrices = {}
    for i in range(len(models)):
            logging.info(f"Computing competencies for client {i}")
            # old version
            # cm = evaluate_competencies(models[i], val_loader, num_classes, True, device)

            # new version
            cm = evaluate_competencies_v2(i, num_classes, total_clients, trainset)
            competency_matrices[i] = cm

    logging.info("Beginning voting")
    y_preds = []
    y_trues = []
    for X, y in testset:
        X = torch.reshape(X, (-1, total_clients, num_classes))
        X = torch.argmax(X, 2)
        for elem_idx in range(X.size(0)):
            relevant_competencies = []
            for client_idx in range(X.size(1)):
                pred_class_by_client = X[elem_idx][client_idx]
                relevant_compt_for_client = [competency_matrices[client_idx][i][pred_class_by_client] for i in range(num_classes)]
                relevant_competencies.append(relevant_compt_for_client)
            
            y_pred_for_elem_idx = get_prediction_using_competency(relevant_competencies, num_classes)
            y_preds.append(y_pred_for_elem_idx)
        y_trues.append(y)
        
    y_preds_np = np.array(y_preds)
    if not require_argmax: # convert to one-hot distribution
        y_preds_np = _one_hot(y_preds_np, num_classes)
    y_trues_np = np.concatenate(y_trues)
    if metric is not None:
        voting_performance = metric(y_trues_np, y_preds_np)
    else:
        voting_performance = accuracy_score(y_trues_np, y_preds_np)
    print(f'==> Voting Performance: {voting_performance}')
    results['voting'] = voting_performance

    #----------------------------------------------------------------------------
    # Linear mapping 
    logging.info("Evaluating Linear Mapping strategy")

    weights = torch.ones(total_clients, requires_grad=True, dtype=torch.float64)
    logging.debug("Size of weights {}".format(weights.size()))
    bias = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
    logging.debug("Size of bias {}".format(bias.size()))

    if training_params:
        lr = training_params["linear_mapping"]['lr']
        epochs = training_params["linear_mapping"]['epochs']
    else: # default values
        lr = 1e-4
        epochs = 1000
    
    if lossfunc is None:
        lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([weights, bias], lr=lr)
    n_batches = len(trainset)

    y_preds = []
    y_trues = []
    best_acc = 0.0
    for e in range(epochs):
        epoch_loss = 0.0
        
        for X, y in trainset:
            optimizer.zero_grad()
            X = torch.reshape(X, (-1, total_clients, num_classes))
            X = torch.swapaxes(X, 1, 2)
            X = X*weights
            y_pred_probits = torch.sum(X, dim=2) + bias
            loss = lossfunc(y_pred_probits, y)
            loss.backward()
            optimizer.step()
            epoch_loss +=  loss.detach().cpu().item()
            
        avg_epoch_loss = epoch_loss / n_batches
        if e%20 == 0:
            logging.debug(f'Running epoch {e+1}/{epochs}')
            logging.debug(f'Average epoch loss {avg_epoch_loss}')
            linear_mapping_performance = _check_accuracy(weights, bias, testset, total_clients, num_classes, metric=metric, require_argmax=require_argmax)
            logging.info(f'Epoch {e+1} Linear Mapping Performance: {linear_mapping_performance}')
            if(linear_mapping_performance > best_acc):
                logging.debug(f'Improved performance from {best_acc:.4f} to {linear_mapping_performance:.4f}')
                best_acc = linear_mapping_performance
                
    print(f'==> Best Linear Mapping Performance: {best_acc}')
    results['linear_mapping'] = best_acc
            
    #----------------------------------------------------------------------------
    # NN Training
    logging.info("Evaluating Neural Network strategy")

    class SmallNN(nn.Module):

        def __init__(self, d=4):
            super().__init__()
            self.fc1 = nn.Linear(total_clients*num_classes, total_clients*d)
            self.fc2 = nn.Linear(total_clients*d, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    # Training params
    if lossfunc is None:
        lossfunc=torch.nn.CrossEntropyLoss()
    
    if training_params:
        lr = training_params["nn"]['lr']
        epochs = training_params["nn"]['epochs']
        f = training_params["nn"]["model"]
    else: # default values
        lr=5e-5
        epochs = 300
        f = SmallNN()
    
    optimizer = Adam(f.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    best_acc, _ = _train_and_evaluate(f, lossfunc, optimizer, scheduler, trainset, testset, epochs, device, \
                                   test_every=5, metric=metric, require_argmax=require_argmax)
    print(f'==> Best NN Performance: {best_acc}')
    results['neural_networks'] = best_acc

    return results