from Defenses.copur import purify_batch
import torch

@torch.no_grad()
def sia_attack_blackbox(data_loader, n_adv, n_clients, n_classes, shuffle_order, amplification=2):
    if n_adv == 0:
        return data_loader

    new_data = []
    nadv_indices = shuffle_order[-n_adv:]

    for data in data_loader:
        x, y = data

        # reshape client logits to [batch, n_clients, n_classes]
        x = x.reshape(-1, n_clients, n_classes)

        # for each adversarial client
        for adv_idx in nadv_indices:
            cur_logits = x[:, adv_idx, :]  # [batch, n_classes]

            # max and min per adversarial client
            max_vals, _ = cur_logits.max(dim=1, keepdim=True)  # [batch, 1]
            min_vals, _ = cur_logits.min(dim=1, keepdim=True)  # [batch, 1]

            # mask true labels to find second best per client
            masked_cur = cur_logits.clone()
            masked_cur[torch.arange(cur_logits.size(0)), y] = float('-inf')
            second_best = masked_cur.argmax(dim=1)  # [batch]

            # initialize adv_logits
            adv_logits = torch.empty_like(cur_logits)

            # case 1: max > 0
            pos_mask = (max_vals > 0).squeeze(1)   # [batch]
            if pos_mask.any():
                adv_logits[pos_mask] = -amplification * max_vals[pos_mask]  # fill with -max
                adv_logits[pos_mask, second_best[pos_mask]] = amplification * max_vals[pos_mask, 0]

            # case 2: max <= 0
            neg_mask = ~pos_mask
            if neg_mask.any():
                adv_logits[neg_mask] = amplification * min_vals[neg_mask]  # fill with min
                adv_logits[neg_mask, second_best[neg_mask]] = -amplification * min_vals[neg_mask, 0]

            # overwrite adversarial client's logits
            x[:, adv_idx, :] = adv_logits

        # flatten back
        new_data.append((x.reshape(-1, n_clients * n_classes), y))

    return new_data

# white-box attack
def sia_attack(data_loader, n_adv, n_clients, n_classes, 
               shuffle_order, model, ae_model, lr, criterion, device,
               initial_iters, final_iters, tau, amplification=2):
    if n_adv == 0:
        return data_loader

    new_data = []
    nadv_indices = shuffle_order[-n_adv:]

    for data in data_loader:
        x, y = data
        outputs = purify_batch(
            data, model, ae_model, lr, criterion, device,
            initial_iters, final_iters, n_clients, n_classes, tau
        )  # shape [batch, n_classes]

        # reshape client logits to [batch, n_clients, n_classes]
        x = x.reshape(-1, n_clients, n_classes)

        # find "second best" class from model outputs
        true_labels = y.clone()
        masked_outputs = outputs.clone()
        masked_outputs[torch.arange(outputs.size(0)), true_labels] = float('-inf')
        second_best = masked_outputs.argmax(dim=1)  # [batch]

        for adv_idx in nadv_indices:
            cur_logits = x[:, adv_idx, :]  # [batch, n_classes]

            # max and min per adversarial client
            max_vals, _ = cur_logits.max(dim=1, keepdim=True)  # [batch,1]
            min_vals, _ = cur_logits.min(dim=1, keepdim=True)  # [batch,1]

            adv_logits = torch.empty_like(cur_logits)

            # case 1: max > 0
            pos_mask = (max_vals > 0).squeeze(1)   # [batch]
            if pos_mask.any():
                adv_logits[pos_mask] = -amplification * max_vals[pos_mask]  # fill all with -max
                adv_logits[pos_mask, second_best[pos_mask]] = amplification * max_vals[pos_mask, 0]

            # case 2: max <= 0
            neg_mask = ~pos_mask
            if neg_mask.any():
                adv_logits[neg_mask] = amplification * min_vals[neg_mask]  # fill all with min
                adv_logits[neg_mask, second_best[neg_mask]] = -amplification * min_vals[neg_mask, 0]

            # overwrite adversarial client's logits
            x[:, adv_idx, :] = adv_logits

        new_data.append((x.reshape(-1, n_clients * n_classes), y))

    return new_data