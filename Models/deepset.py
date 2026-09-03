import torch.nn as nn
import torch
from .static_aggs import bpda_median, bpda_trimmed_mean

### Without masking
class DeepSet_base(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).sum(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        return X

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128, output_prob=False):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.output_prob = output_prob
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)
        
        # Apply the mask: [batch_size, set_size, feature_dim]
        masked_encoded = encoded * mask.unsqueeze(-1)  # Mask out padded entries
        
        # Sum along the set dimension (ignoring padded parts)
        aggregated = masked_encoded.sum(dim=1)  # Sum valid elements
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)

        if self.output_prob:
            # take softmax over the output
            output = torch.softmax(output, dim=-1)
        
        return output

class DeepSet_M(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128):
        super(DeepSet_M, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)
        
        # Apply the mask: [batch_size, set_size, feature_dim]
        masked_encoded = encoded * mask.unsqueeze(-1)  # Mask out padded entries
        
        # Sum along the set dimension (ignoring padded parts)
        aggregated = masked_encoded.mean(dim=1)  # Mean valid elements
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        
        return output

class DeepSet_Median(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128):
        super(DeepSet_Median, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)

        # assert all values are 1 in mask
        assert torch.all((mask == 1)), "Error: DeepSet_Median cannot be used with masking."
        
        # Apply the mask: [batch_size, set_size, feature_dim]
        masked_encoded = encoded * mask.unsqueeze(-1)  # Mask out padded entries
        
        # Sum along the set dimension (ignoring padded parts)
        aggregated = masked_encoded.median(dim=1).values  # Median valid elements
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        
        return output

class DeepSet_Median2(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128):
        super(DeepSet_Median2, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)

        # assert all values are 1 in mask
        assert torch.all((mask == 1)), "Error: DeepSet_Median cannot be used with masking."
        
        # Apply the mask: [batch_size, set_size, feature_dim]
        masked_encoded = encoded * mask.unsqueeze(-1)  # Mask out padded entries
        
        # Sum along the set dimension (ignoring padded parts)
        aggregated = bpda_median(masked_encoded).squeeze()  # Median valid elements
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        
        return output

class DeepSet_TM2(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128, trim_ratio=0.1, output_prob=False):
        super(DeepSet_TM2, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.trim_ratio = trim_ratio
        self.output_prob = output_prob
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)
                
        # Sum along the set dimension (ignoring padded parts)
        aggregated = bpda_trimmed_mean(encoded, self.trim_ratio).squeeze()  # Trimmed mean valid elements
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        
        return output

### This is trimmed mean    
class DeepSet_TM(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output=1, dim_hidden=128, trim_ratio=0.1, output_prob=False):
        super(DeepSet_TM, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.trim_ratio = trim_ratio
        self.output_prob = output_prob
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X, mask):
        # Apply the encoding layer
        encoded = self.enc(X)  # shape: [batch_size, set_size, feature_dim]
        
        # Apply the mask: zeros out padded entries
        masked_encoded = encoded * mask.unsqueeze(-1)  # shape: [B, N, D]
        
        B, _, _ = masked_encoded.shape
        aggregated_list = []
        
        # Process each batch item individually
        for i in range(B):
            # Get the valid entries for this sample (mask should be 1 for valid, 0 for padded)
            valid_mask = mask[i].bool()  # shape: [N]
            valid_elements = masked_encoded[i][valid_mask]  # shape: [num_valid, D]
            num_valid = valid_elements.shape[0]
            
            if num_valid > 0:
                # Determine how many elements to trim from each end
                k = int(num_valid * self.trim_ratio)
                # if i == 0: logging.debug(f"Trimming {k} elements from each end of the set.")
                
                # If there are enough elements to trim, perform trimming per feature.
                # Sorting is done along dim=0 (i.e. over the set elements) for each feature independently.
                if num_valid > 2 * k:
                    sorted_valid, _ = torch.sort(valid_elements, dim=0)
                    trimmed = sorted_valid[k:num_valid - k]
                else:
                    raise ValueError(f"Not enough valid elements after trimming: num_valid ({num_valid}) <  2 * k ({2 * k})")
                
                # Sum the remaining (trimmed) values over the set dimension
                aggregated_list.append(trimmed.mean(dim=0))
            else:
                raise ValueError("No valid elements found in the set.")
        
        # Stack the aggregated representations back into a batch
        aggregated = torch.stack(aggregated_list, dim=0)  # shape: [B, D]
        
        # Decode the aggregated representation
        output = self.dec(aggregated).reshape(-1, self.num_outputs, self.dim_output).squeeze(-1)
        
        if self.output_prob:
            # take softmax over the output
            output = torch.softmax(output, dim=-1)

        return output