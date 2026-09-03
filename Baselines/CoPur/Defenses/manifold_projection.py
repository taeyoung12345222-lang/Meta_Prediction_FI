import torch
import numpy as np
from sklearn.metrics import accuracy_score

@torch.no_grad()
def manifold_projection(test_loader, model, ae_model, device, num_clients, num_classes):
    model.eval()
    model = model.to(device)

    ae_model.eval()
    ae_model = ae_model.to(device)

    y_preds = []
    y_trues = []

    for (data, target) in test_loader:
        data = data.to(device)
        
        output = ae_model(data)
        final_output = model(output)

        y_preds.append(torch.argmax(final_output, dim=1).cpu())
        y_trues.append(target.cpu())

    y_preds_np = np.concatenate(y_preds)
    y_trues_np = np.concatenate(y_trues)

    acc1 = accuracy_score(y_trues_np, y_preds_np)
    
    return acc1