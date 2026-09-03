import torch.nn as nn
import torch
import logging

# NN model that just returns the average of the logits
class F_Avg(nn.Module):
    def __init__(self, n_classes):
        super(F_Avg, self).__init__()
        self.n_classes = n_classes

    def forward(self, x, mask):
        # input is a tensor of shape (batch_size, n_clients, n_classes)
        return x.mean(dim=1).squeeze()

# Median
class F_Median(nn.Module):
    def __init__(self, n_classes, output_prob=False):
        super(F_Median, self).__init__()
        self.n_classes = n_classes
        self.output_prob = output_prob

    def forward(self, x, mask):
        # input is a tensor of shape (batch_size, n_clients, n_classes)
        output = x.median(dim=1).values.squeeze()
        if self.output_prob:
            return output / output.sum(dim=1, keepdim=True)
        else:
            return output

class BPDAMedian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # save input shape for backward
        ctx.input_shape = x.shape  
        median_vals, _ = torch.median(x, dim=1)  # [batch, n_classes]
        return median_vals

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [batch, n_classes]
        # need to return gradient w.r.t x: [batch, n_clients, n_classes]

        batch, n_clients, n_classes = ctx.input_shape

        # BPDA trick: approximate as identity -> just broadcast grad_output to all clients
        grad_input = grad_output.unsqueeze(1).expand(batch, n_clients, n_classes)
        return grad_input

def bpda_median(x):
    return BPDAMedian.apply(x)

class F_Median2(nn.Module):
    def __init__(self, n_classes, output_prob=False):
        super(F_Median2, self).__init__()
        self.n_classes = n_classes
        self.output_prob = output_prob

    def forward(self, x, mask):
        # input is a tensor of shape (batch_size, n_clients, n_classes)
        output =  bpda_median(x).squeeze()
        if self.output_prob:
            return output / output.sum(dim=1, keepdim=True)
        else:
            return output
    
# Geometric Median
class F_Geo_Median(nn.Module):
    def __init__(self, n_classes, output_prob=False, max_iter=20, tol=1e-6):
        super(F_Geo_Median, self).__init__()
        self.n_classes = n_classes
        self.output_prob = output_prob
        self.max_iter = max_iter # For Weiszfeld's algorithm
        self.tol = tol

    def forward(self, x, mask):
        # input is a tensor of shape (batch_size, n_clients, n_classes)
        y = x.mean(dim=1)  # Initialize y as the mean of the points

        for _ in range(self.max_iter):
            distances = torch.norm(x - y.unsqueeze(1), dim=2) # shape: (batch_size, n_clients)
            w = 1. / (distances + 1e-8)

            w = w / w.sum(dim=1, keepdim=True) # normalize
            y_new = (w.unsqueeze(2) * x).sum(dim=1) # shape: (batch_size, n_classes)

            if torch.max(torch.norm(y_new - y)) < self.tol:
                break

            y = y_new

        if self.output_prob:
            return y / y.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            return y

# Trimmed mean
class F_TM(nn.Module):
    def __init__(self, n_classes, trim_ratio=0.1, output_prob=False):
        super(F_TM, self).__init__()
        self.n_classes = n_classes
        self.trim_ratio = trim_ratio
        self.output_prob = output_prob

    def forward(self, x, mask):
        # input is a tensor of shape (batch_size, n_clients, n_classes)
        n_clients = x.shape[1]
        n_trim = int(n_clients * self.trim_ratio)
        logging.debug(f"Trimming {n_trim} clients on each side out of {n_clients}")
        x_sorted, _ = x.sort(dim=1)
        output = x_sorted[:, n_trim:-n_trim].mean(dim=1).squeeze()
        if self.output_prob:
            return output / output.sum(dim=1, keepdim=True)
        else:
            return output

class BPDATrimmedMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, trim_ratio):
        ctx.input_shape = x.shape
        ctx.trim_ratio = trim_ratio

        n_clients = x.shape[1]
        n_trim = int(n_clients * trim_ratio)

        # Sort across clients
        x_sorted, _ = torch.sort(x, dim=1)

        # Drop extremes
        if n_trim > 0:
            trimmed = x_sorted[:, n_trim:-n_trim]
        else:
            trimmed = x_sorted

        output = trimmed.mean(dim=1)  # [batch, n_classes]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [batch, n_classes]
        batch, n_clients, n_classes = ctx.input_shape

        # BPDA trick: approximate trimmed mean as identity
        grad_input = grad_output.unsqueeze(1).expand(batch, n_clients, n_classes)

        # two outputs in forward: x and trim_ratio → return grad for each
        return grad_input, None


def bpda_trimmed_mean(x, trim_ratio=0.1):
    return BPDATrimmedMean.apply(x, trim_ratio)


class F_TM2(nn.Module):
    def __init__(self, n_classes, trim_ratio=0.1, output_prob=False):
        super(F_TM2, self).__init__()
        self.n_classes = n_classes
        self.trim_ratio = trim_ratio
        self.output_prob = output_prob

    def forward(self, x, mask):
        # input: [batch, n_clients, n_classes]
        output = bpda_trimmed_mean(x, self.trim_ratio).squeeze()
        if self.output_prob:
            return output / output.sum(dim=1, keepdim=True)
        else:
            return output