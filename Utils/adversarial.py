import torch
import logging

def norm_normalize(x, norm_value, eps=1e-8):
    """
    if normalization_type == 'fix-norm':
        Normalizes tensor x along its last dimension to have a norm of norm_value,
        but only if its current norm exceeds norm_value.
    elif normalization_type == 'range':
        Normalizes tensor x along its last dimension to have a norm in the range [-norm_value, norm_value].
    
    Args:
        x (torch.Tensor): The input tensor.
        norm_value (float): The threshold norm value.
        eps (float): Small epsilon to avoid division by zero.
    
    Returns:
        torch.Tensor: The conditionally normalized tensor.
    """
    # Compute the L2 norm along the last dimension with keepdim=True for broadcasting.
    current_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    
    # Compute the scaling factor: if current_norm > norm_value, use norm_value/current_norm, else 1.
    scale = torch.where(current_norm > norm_value, norm_value / (current_norm + eps), torch.ones_like(current_norm))

    # Scale x accordingly.
    return x * scale
    
def range_normalize(dataset, norm_value, eps=1e-8, max_vals=None, min_vals=None):

    if max_vals is None: 
        max_vals = [torch.max(x, dim=-1, keepdim=True).values for x, _ in dataset]
        max_vals = torch.cat(max_vals, dim=-1)
        max_vals = max_vals.max(dim=-1, keepdim=True).values

    if min_vals is None:
        min_vals = [torch.min(x, dim=-1, keepdim=True).values for x, _ in dataset]
        min_vals = torch.cat(min_vals, dim=-1)
        min_vals = min_vals.min(dim=-1, keepdim=True).values
    
    dataset = [((2 * norm_value * (x - min_vals) / (max_vals - min_vals + eps)) - norm_value, y) for x, y in dataset]

    return dataset, max_vals, min_vals

def project_onto_simplex(v, eps=1e-8):
    """
    Projects each row of v onto the probability simplex using the sorting-based algorithm.
    
    Args:
        v (torch.Tensor): The input tensor of shape (..., n_classes), where each row is projected.
        eps (float): A small value to avoid division issues.

    Returns:
        torch.Tensor: The projected tensor of the same shape as v.
    """
    sorted_v, _ = torch.sort(v, descending=True, dim=-1)
    cumulative_sum = torch.cumsum(sorted_v, dim=-1)
    rho = torch.arange(1, v.shape[-1] + 1, device=v.device).float()
    
    # Compute the threshold index
    condition = sorted_v - (cumulative_sum - 1) / rho
    condition = condition > 0
    rho_index = condition.sum(dim=-1, keepdim=True) - 1  # Get the last True index for each row
    
    # Compute the threshold value theta
    theta = (cumulative_sum.gather(-1, rho_index) - 1) / (rho_index + 1).float()
    
    # Apply the projection
    return torch.clamp(v - theta, min=0)

def adversarial_attack_batch(x, mask, y, f, device, n_adv, loss_fn, n_iter=10, 
                             alpha=0.01, eps=0.1):
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)

    # append n_adv rows to each x which will be the adversarial perturbations
    # x has shape [batch_size, n_clients, n_classes]
    # new shape: [batch_size, n_clients + n_adv, n_classes]
    adv_part = torch.rand_like(x[:, :n_adv]).requires_grad_(True)
    
    # update mask
    mask = torch.cat([mask, torch.ones_like(mask[:, :n_adv])], dim=1).clone().detach().requires_grad_(False)

    logging.debug(f'Adversarial attack on a new batch')
    for i in range(n_iter):
        # logging.debug(f'Iteration {i}/{n_iter}')
        f.zero_grad()

        # take softmax as a part of the computation graph
        x_adv = torch.cat([x, adv_part.softmax(dim=-1)], dim=1)

        loss = loss_fn(f(x_adv, mask), y)
        if i % 10 == 0: logging.debug(f'Loss at {i}/{n_iter}: {loss.item()}')
        loss.backward()
        
        # update adversarial examples using the sign of the gradient
        adv_part.data += alpha * adv_part.grad.sign()

        adv_part.grad.zero_()
    
    ### turn into probability vector after all iterations
    adv_part = adv_part.softmax(dim=-1)
    
    x_adv = torch.cat([x, adv_part], dim=1).clone().detach().requires_grad_(False)

    # move to cpu
    x_adv = x_adv.cpu()
    mask = mask.cpu()

    return x_adv, mask

def adversarial_attack_batch_inplace(x, mask, y, f, device, loss_fn, 
                                 n_iter=10, alpha=0.01, eps=0.1):
    """
    Performs adversarial attack on clients marked as adversaries (mask==0).
    
    Args:
        x (Tensor): [batch_size, num_clients, num_classes] input probability vectors.
        mask (Tensor): [batch_size, num_clients] binary mask. 
                       Clients with 0 are adversarial and will be perturbed.
        y (Tensor): Labels.
        f (Module): Aggregator model.
        device (torch.device): Device.
        loss_fn (function): Loss function.
        n_iter (int): Number of gradient ascent iterations.
        alpha (float): Step size for update.
        eps (float): (Unused here but kept for compatibility.)
        normalize (bool): Whether to apply additional projection.
        norm_value (float): Value used in normalization.
        normalization_type (str): Either 'fix-norm' or 'range'.
        
    Returns:
        x_adv (Tensor): The updated batch with adversarial entries replaced.
        mask (Tensor): Unchanged mask.
    """
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)

    # Identify adversarial positions: those where mask == 0.
    adv_mask = (mask == 0)  # shape: [batch_size, num_clients]
    total_adv = adv_mask.sum()  # total number of adversarial entries
    logging.debug(f'Found {total_adv} adversarial clients in the batch.')

    # If no adversarial clients are found, return original inputs.
    if total_adv == 0:
        return x.cpu(), mask.cpu()    

    # Randomly initialize logits for adversarial positions.
    # Note: We work in logit space and use softmax later to ensure probabilities.
    adv_logits = torch.rand((total_adv, x.shape[-1]), device=device, requires_grad=True)
    all_ones_mask = torch.ones_like(mask)
    
    logging.debug('Adversarial attack on a new batch')
    for i in range(n_iter):
        # logging.debug(f'Iteration {i}/{n_iter}')
        f.zero_grad()

        # Create a fresh copy of x
        x_adv = x.clone()

        # For all adversarial positions, replace the entry with softmax(adv_logits)
        x_adv[adv_mask] = adv_logits.softmax(dim=-1)
        
        loss = loss_fn(f(x_adv, all_ones_mask), y)
        if i % 5 == 0:
            logging.debug(f'Loss at {i}/{n_iter}: {loss.item()}')
        loss.backward()
        
        # Update only the adversarial logits with a gradient ascent step.
        adv_logits.data += alpha * adv_logits.grad.sign()
                
        adv_logits.grad.zero_()

    # Finalize adversarial examples: apply softmax so they remain valid probability vectors.
    x_adv = x.clone()
    x_adv[adv_mask] = adv_logits.softmax(dim=-1).clone().detach()

    # Move to CPU before returning.
    return x_adv.cpu(), all_ones_mask.cpu()

def adversarial_attack(f, test_loader, device, n_adv, loss_fn, n_iter=10, alpha=0.01, eps=0.1):
    f.eval()
    f = f.to(device)

    # generate adversarial examples
    adv_examples = []; batch_idx = 0
    for data, mask, target in test_loader:
        batch_idx += 1
        logging.debug(f'Generating adversarial examples for {batch_idx}/{len(test_loader)}')
        adv_data, adv_mask = adversarial_attack_batch(data, mask, target, f, device, n_adv, loss_fn, \
                                                      n_iter, alpha, eps)
        adv_examples.append((adv_data, adv_mask, target))        

    return adv_examples

@torch.no_grad()
def sia_attack(f, test_loader, device, n_adv):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating SIA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        x_adv = x.clone().detach()
        mask_adv = mask.clone().detach()
        
        ### shape of x is [batch_size, num_clients, num_classes]
        y_pred = f(x, mask)
        ### find top2 classes
        top2_indices = torch.topk(y_pred, 2, dim=-1).indices
        ### find the top class that is not the true class
        top1_not_true_index = torch.where(top2_indices[:, 0] == y, top2_indices[:, 1], top2_indices[:, 0])
        ### expand for n_adv
        top1_not_true_index = top1_not_true_index.unsqueeze(-1).expand(-1, n_adv)
        ### set the top class that is not the true class to 1
        x_adv[:, -n_adv:] = x_adv[:, -n_adv:].scatter(-1, top1_not_true_index.unsqueeze(-1), 1)

        # move to cpu
        x_adv = x_adv.cpu()
        mask_adv = mask_adv.cpu()
        y = y.cpu()

        adv_examples.append((x_adv, mask_adv, y))
    
    return adv_examples

@torch.no_grad()
def sia_attack_blackbox_collude(f, test_loader, device, n_adv):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating SIA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        x = x.to(device)
        mask = mask.to(device)

        B, _, C = x.shape  # batch_size, num_clients, num_classes

        x_adv = x.clone().detach()
        mask_adv = mask.clone().detach()
        
        ### shape of x is [batch_size, num_clients, num_classes]
        y_pred = f(x[:,-n_adv:], mask[:,-n_adv:]) ### aggregate only over the byzantine nodes
        ### find top2 classes
        top2_indices = torch.topk(y_pred, 2, dim=-1).indices # shape: [B, 2]
        second_best = top2_indices[:, 1] # shape: [B]

        one_hot = torch.zeros(B, C, device=device)
        one_hot.scatter_(1, second_best.unsqueeze(1), 1) # shape: [B, C]

        ### set the top class that is not the true class to 1
        x_adv[:, -n_adv:] = one_hot.unsqueeze(1).expand(-1, n_adv, -1) # shape: [B, n_adv, C]

        # move to cpu
        x_adv = x_adv.cpu()
        mask_adv = mask_adv.cpu()

        adv_examples.append((x_adv, mask_adv, y))
    
    return adv_examples

@torch.no_grad()
def sia_attack_blackbox_collude_batch(f, x, mask, device):
    x = x.to(device)
    mask = mask.to(device)

    B, E, C = x.shape  # batch_size, num_clients, num_classes

    x_adv = x.clone().detach()
    
    # Boolean mask: 1 if malicious, 0 if honest
    mask_mal_bool = ~mask.bool()  # shape: [B, E]
    mask_malicious_f = mask_mal_bool.float()  # for aggregation

    # Predict using only malicious client votes
    y_pred = f(x, mask_malicious_f)  # shape: [B, C]

    # Get second highest predicted class
    top2_indices = torch.topk(y_pred, 2, dim=-1).indices  # [B, 2]
    second_best = top2_indices[:, 1]                      # [B]

    # Create one-hot vote for second best class
    one_hot = torch.zeros(B, C, device=device)
    one_hot.scatter_(1, second_best.unsqueeze(1), 1)  # [B, C]

    # Expand for all malicious clients
    one_hot_expanded = one_hot.unsqueeze(1).expand(-1, E, -1)  # [B, E, C]
    mask_malicious_expanded = mask_mal_bool.unsqueeze(-1)     # [B, E, 1]

    # Apply adversarial vote to malicious clients
    x_adv = x_adv * (~mask_malicious_expanded) + one_hot_expanded * mask_malicious_expanded

    return x_adv.cpu(), torch.ones_like(mask).cpu()

@torch.no_grad()
def sia_attack_blackbox(test_loader, device, n_adv):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating SIA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        x = x.to(device)  # shape: (B, E, C)
        mask = mask.to(device)
        y = y.to(device)

        B, E, _ = x.shape
        x_adv = x.clone().detach()
        mask_adv = mask.clone().detach()

        # Get indices of last n_adv experts
        adv_expert_indices = torch.arange(E - n_adv, E, device=device)  # shape: (n_adv,)

        # Select logits for adversarial experts: shape (B, n_adv, C)
        adv_logits = x_adv[:, adv_expert_indices, :]

        # Get top-2 indices per expert per sample: shape (B, n_adv, 2)
        top2 = torch.topk(adv_logits, k=2, dim=2).indices
        second_best_indices = top2[:, :, 1]  # shape: (B, n_adv)

        # Create new logits: all zeros, 1 at second_best_indices
        new_adv_logits = torch.zeros_like(adv_logits)  # (B, n_adv, C)
        row_idx = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
        expert_idx = torch.arange(n_adv, device=device).unsqueeze(0)  # (1, n_adv)

        # Fill in the one-hot at the second-best positions
        new_adv_logits[row_idx, expert_idx, second_best_indices] = 1.0

        # Replace adversarial expert logits in x_adv
        x_adv[:, adv_expert_indices, :] = new_adv_logits

        adv_examples.append((x_adv, mask_adv, y))

    return adv_examples

@torch.no_grad()
def sia_attack_blackbox_batch(x, mask, device):
    """
    Apply Strongest Inverted Attack (SIA) on a batch of expert responses in a vectorized way.

    Args:
        x (torch.Tensor): Tensor of shape (B, n_clients, n_classes), each entry is a probability vector.
        mask (torch.Tensor): Tensor of shape (B, n_clients), binary values indicating malicious clients (1 = malicious).
        device (torch.device): The torch device for tensor allocations.

    Returns:
        torch.Tensor: Adversarial version of x with SIA applied to malicious clients.
    """
    x = x.to(device)
    mask = mask.to(device)
    x_adv = x.clone()

    # Get top-2 indices per client (shape: B x E x 2)
    top2 = torch.topk(x, k=2, dim=2).indices.to(device)

    # Get the second-best index (B x E)
    second_best_idx = top2[:, :, 1]

    # Find indices where mask == 0 (malicious clients)
    malicious_indices = ~mask.bool()  # shape: (B, E)

    # Create a zero tensor and fill second-best index with 1
    one_hot = torch.zeros_like(x, device=device)

    # Flatten to get malicious (b, e) coordinates
    flat_idx = malicious_indices.nonzero(as_tuple=False)  # shape: (N, 2)
    b_idx = flat_idx[:, 0]
    e_idx = flat_idx[:, 1]
    c_idx = second_best_idx[b_idx, e_idx]

    # Set one-hot entry to 1 for second-best class
    one_hot[b_idx, e_idx, c_idx] = 1.0

    # Replace malicious entries in x_adv
    x_adv[malicious_indices] = one_hot[malicious_indices]

    return x_adv.cpu(), torch.ones_like(mask).cpu()

@torch.no_grad()
def loss_maximization_attack_batch(x, mask, f, device, n_adv):
    """
    Performs loss maximization attack by setting adversarial clients' votes to the least likely class.
    Args:
        x (Tensor): [batch_size, num_clients, num_classes] logits.
        mask (Tensor): [batch_size, num_clients] binary mask. 
                       Clients with 0 are adversarial and will be perturbed.
        y (Tensor): Labels.
        f (Module): Aggregator model.
        device (torch.device): Device.

    Returns:
        x_adv (Tensor): The updated batch with adversarial entries replaced.
        mask (Tensor): Unchanged mask.
    """
    x = x.to(device)
    mask = mask.to(device)

    x_adv = x.clone().detach()
    B, _, C = x.shape  # batch_size, num_clients, num_classes

    # compute aggregated honest prediction Ȳ_H(x) using aggregator f
    Y_H = f(x, mask)  # shape: [B, C]

    # choose least likely class
    least_likely_idx = torch.argmin(Y_H, dim=-1)  # shape: [B]

    # create one-hot for chosen class and expand to all malicious clients
    one_hot = torch.zeros(B, C, device=device)
    one_hot.scatter_(1, least_likely_idx.unsqueeze(1), 1)

    x_adv[:, -n_adv:] = one_hot.unsqueeze(1).expand(-1, n_adv, -1) # shape: [B, n_adv, C]

    return x_adv.cpu() , torch.ones_like(mask).cpu()

@torch.no_grad()
def loss_maximization_attack(f, test_loader, device, n_adv):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating LMA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        adv_data, adv_mask = loss_maximization_attack_batch(x, mask, f, device, n_adv)
        adv_examples.append((adv_data, adv_mask, y))

    return adv_examples

@torch.no_grad()
def compute_similarity_matrix(model, loader, device):
    """
    Computes the  class similarity matrix S using a pre-trained model.
    Args:
        model: pretrained classifier returning probabilities (shape: (batch_size, n_classes))
        loader: Dataloader yielding (inputs, labels, mask)
        device: torch device
    
    Returns:
        S: class similarity matrix of shape (n_classes, n_classes)
    """
    model.eval()
    predictions = []
    for inputs, mask, _ in loader:
        inputs = inputs.to(device)

        probs = model(inputs, mask)  # shape: (batch_size, n_classes)
        predictions.append(probs.cpu())

    predictions = torch.cat(predictions, dim=0)  # shape: (N, n_classes)
    centered_predictions = predictions - predictions.mean(dim=0).unsqueeze(0)  # shape: (N, n_classes)

    S = centered_predictions.T @ centered_predictions / centered_predictions.shape[0]  # shape: (n_classes, n_classes)

    return S

@torch.no_grad()
def class_prior_attack_batch(x, mask, f, device, n_adv, S):
    """
    Performs class prior attack by setting adversarial clients' votes to the least similar class based on a similarity matrix.
    """
    x = x.to(device)
    mask = mask.to(device)

    x_adv = x.clone().detach()
    B, _, C = x.shape  # batch_size, num_clients, num_classes

    # compute aggregated honest prediction Ȳ_H(x) using aggregator f
    Y_H = f(x, mask)  # shape: [B, C]

    # choose most likely class
    most_likely_idx = torch.argmax(Y_H, dim=-1)  # shape: [B]

    # for each class i find j = argmin_j S[i, j] (least similar class to i)
    S_rows = S[most_likely_idx] # shape: [B, C]
    least_similar = S_rows.argmin(dim=-1) # shape: [B]

    # create one-hot for chosen class and expand to all malicious clients
    one_hot = torch.zeros(B, C, device=device)
    one_hot.scatter_(1, least_similar.unsqueeze(1), 1)
    
    x_adv[:, -n_adv:] = one_hot.unsqueeze(1).expand(-1, n_adv, -1) # shape: [B, n_adv, C]

    return x_adv.cpu(), torch.ones_like(mask).cpu()

@torch.no_grad()
def class_prior_attack(f, test_loader, device, n_adv, S):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating CPA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        adv_data, adv_mask = class_prior_attack_batch(x, mask, f, device, n_adv, S)
        adv_examples.append((adv_data, adv_mask, y))

    return adv_examples

@torch.no_grad()
def impersonation_attack_batch(x, mask, device, n_adv):
    """
    Performs impersonation attack by mimicking the honest client whose logits are farthest away from the others (L2 distance).
    """
    x = x.to(device)
    mask = mask.to(device)

    x_adv = x.clone().detach() 
    B, _, C = x.shape # shape: [B, E, C]

    honest_clients = x[:, :-n_adv, :]  # shape: [B, H, C] with H = E - n_adv
    # pairwise L2 distances between honest clients
    distances = torch.cdist(honest_clients, honest_clients, p=2) # shape: [B, H, H]
    sum_distances = distances.sum(dim=-1)  # shape: [B, H]
    farthest_idx = torch.argmax(sum_distances, dim=1)  # shape: [B]

    farthest_idx_expanded = farthest_idx.view(B, 1, 1).expand(-1, 1, C)  # shape: [B, 1, C]
    farthest_logits = honest_clients.gather(1, farthest_idx_expanded) # shape: [B, 1, C]

    x_adv[:, -n_adv:, :] = farthest_logits.expand(-1, n_adv, -1) # shape: [B, n_adv, C]

    return x_adv.cpu(), torch.ones_like(mask).cpu()

@torch.no_grad()
def impersonation_attack(f, test_loader, device, n_adv):
    adv_examples = []
    for x, mask, y in test_loader:
        logging.debug(f'Generating IA adversarial examples for batch {len(adv_examples) + 1}/{len(test_loader)}')
        adv_data, adv_mask = impersonation_attack_batch(x, mask, device, n_adv)
        adv_examples.append((adv_data, adv_mask, y))

    return adv_examples

def Carlini_Wagner_loss(logits, labels, targeted=False, confidence=0.1, input_is_prob=True):
    """
    Compute the Carlini-Wagner loss for adversarial attacks (for gradient ascent).
    
    Args:
        logits (torch.Tensor): The input logits of shape (batch_size, num_classes).
        labels (torch.Tensor): The true labels (or target labels for targeted attacks) of shape (batch_size,).
        targeted (bool): Whether to perform a targeted attack.
        confidence (float): The confidence margin (κ).
    
    Returns:
        torch.Tensor: The computed loss (negated for gradient ascent).
    """
    if not input_is_prob:
        logits = torch.softmax(logits, dim=-1)

    batch_size, num_classes = logits.shape

    # check that the logits are in fact probabilities that sum to one
    assert torch.allclose(logits.sum(dim=-1), torch.ones(batch_size, device=logits.device)), \
        f'Logits do not sum to one: {logits.sum(dim=-1)}'
    
    # Get logit of the true (or target) class
    correct_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)

    # Get highest logit for incorrect classes
    mask = torch.ones_like(logits).scatter(1, labels.view(-1, 1), 0)  # Mask out true label
    other_logits = logits[mask.bool()].view(batch_size, num_classes - 1)  # Extract non-correct logits
    max_other_logits, _ = other_logits.max(dim=1)  # Get max logit of incorrect classes

    # Compute loss
    if targeted:
        # Targeted attack: make target class more confident
        loss = max_other_logits - correct_logits + confidence
    else:
        # Untargeted attack: reduce confidence in the original class
        loss = correct_logits - max_other_logits + confidence

    # Ensure non-negative loss
    loss = loss.clamp(min=0)

    # **Negate the loss** for gradient ascent
    return -loss.mean()

### Cross-Entropy loss with probabilities
def cross_entropy_from_probs(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Cross-entropy loss where `input` is a probability vector (after softmax).
    
    Args:
        input: Tensor of shape (batch_size, num_classes), must be probabilities (sums to 1).
        target: Tensor of shape (batch_size, 1) or (batch_size,), with class indices.
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Scalar loss or loss per example depending on `reduction`.
    """
    if target.dim() == 2:
        target = target.squeeze(1)
    
    # Gather log probs for the correct class index
    log_input = input.clamp(min=1e-9).log()  # avoid log(0)
    losses = -log_input[torch.arange(input.size(0)), target]

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError(f"Invalid reduction: {reduction}")