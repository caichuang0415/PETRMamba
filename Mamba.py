import torch
from mamba_ssm import Mamba

# an example use of Mamba
# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape

class PETRMamba:
    """A substitution of original PETRMultiheadFlashAttention.
    Args:
        d_model (int): Model dimension
        d_state (int): SSM state expansion factor
        d_conv (int): Local convolution width
        expand (int): Block expansion factor
    """
    def __init__(self, d_model, d_state, d_conv, expand, **kwargs):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        
    def merge(self, memory, query, attn_mask=None, diffrent_src=True):
        # merge memory and query as one sequence
        ret = torch.empty_like(query)
        return ret
        
    def forward(self, 
                memory, 
                query,
                attn_mask=None,
                diffrent_src=True):
        """Forward function for `PETRMamba`.
        Args:
            query (Tensor): [num_queries, bs, d_model] or [bs, num_queries, d_model]
            memory (Tensor): [num_memories, bs, d_model] or [bs, num_memories, d_model]
            attn_mask (Tensor): ByteTensor mask with shape [num_queries, num_memories]
            diffrent_src (Bool): if diffrent_src is True, that means memory and query come 
            from the same source and thus the PETRMamba should do a self-attention like operation
            Else, the PETRMamba should do a cross-attention like operation
        Returns:
            Tensor: the same shape as query
        """
        merged_sequence = self.merge(memory, query, attn_mask, diffrent_src)
        return self.mamba(merged_sequence)