import torch
import logging
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from Utils.metrics import Meter

def purify_batch(batch_data, model, ae_model, lr, criterion, device,
                 initial_iters, final_iters, num_clients, num_classes, tau):
    model.eval()
    model = model.to(device)

    ae_model.eval()
    ae_model = ae_model.to(device)
    
    H, _ = batch_data
    H = H.to(device)
    H_split = torch.split(H, num_classes, dim=1)

    with torch.no_grad():
        output = ae_model(H)

    L = output.clone().detach().requires_grad_()
    optimizer = Adam([L], lr=lr)
    losses = Meter(ptag='Loss')
    
    for i_iter in range(initial_iters):
        output = ae_model(L)
        output_split = torch.split(output, num_classes, dim=1)

        loss = 0.0
        for i_c in range(num_clients):
            loss += torch.sqrt(criterion(output_split[i_c], H_split[i_c]))

        # backward pass
        loss.backward()

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), H.size(0))

        if i_iter % 5 == 0:
            logging.debug('Initial step {ep} Loss {loss.avg:.5f}'.format(ep=i_iter, loss=losses))

    L2 = L.clone().detach().requires_grad_()
    optimizer = Adam([L2], lr=lr)
    losses2 = Meter(ptag='Loss')

    for f_iter in range(final_iters):

        output = ae_model(L2) # [batch_size, num_clients * num_classes]
        output_split = torch.split(output, num_classes, dim=1)
        L2_split = torch.split(L2, num_classes, dim=1)

        loss2 = 0.0
        for i_c in range(num_clients):
            loss2 += tau * torch.sqrt(criterion(output_split[i_c], L2_split[i_c]))
        
        for i_c in range(num_clients):
            loss2 += torch.sqrt(criterion(H_split[i_c], L2_split[i_c]))
        
        loss2.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses2.update(loss2.item(), H.size(0))

        if f_iter % 5 == 0:
            logging.debug('Final step {ep} Loss {loss.avg:.5f}'.format(ep=f_iter, loss=losses2))
    
    final_output = model(output)

    return final_output

def purify(test_loader, model, ae_model, lr, criterion, device, initial_iters, 
           final_iters, num_clients, num_classes, tau):

    y_preds = []
    y_trues = []

    for batch_idx, batch_data in enumerate(test_loader):
        logging.debug(f"Batch {batch_idx}/{len(test_loader)}")
        _, y = batch_data
        final_output = purify_batch(batch_data, model, ae_model, lr, criterion,
                                    device, initial_iters, final_iters, num_clients, 
                                    num_classes, tau)


        y_preds.append(torch.argmax(final_output, dim=1).cpu())
        y_trues.append(y.cpu())

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)

    acc1 = accuracy_score(y_trues_np, y_preds_np)
    
    return acc1