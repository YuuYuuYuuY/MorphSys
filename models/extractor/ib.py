import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .bfe import PathAdd, PathLstm


class InterBranch(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, path:str="add", direction:str="out"):
        super(InterBranch,self).__init__()

        if path == "add":
            self.bfe = PathAdd()
        elif path == "lstm":
            self.bfe = PathLstm(feat_size=embed_dim, hidden_size=embed_dim)
        else:
            self.bfe = PathAdd()

        self.direction = direction

        self.inter_branch_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
    
    
    def forward(self, g:dgl.DGLGraph, max_len:int, offset_leaf):
        # propagate
        if self.direction == "out":
            dgl.prop_nodes_topo(
                g,
                message_func=self.bfe.message_func,
                reduce_func=self.bfe.reduce_func,
                apply_node_func=self.bfe.apply_node_func,
                reverse=True
            )
        else:
            dgl.prop_nodes_topo(
                g,
                message_func=self.bfe.message_func,
                reduce_func=self.bfe.reduce_func,
                apply_node_func=self.bfe.apply_node_func
            )

        temp_data = g.ndata.pop("feats")
        # 通过 offset_hop 看成一个不定长序列, 即tokenized
        branch_feats = []
        branch_mask = []
        # max_len = batch.max_len
        for i in offset_leaf:
            # get branch feature
            branch_feats.append(temp_data[i])
            # add padded mask
            temp_mask = torch.ones(len(i), len(i))
            padded_mask = F.pad(temp_mask, (0, max_len - len(i), 0, max_len - len(i)), value=0)
            branch_mask.append(padded_mask)
        #     print(f"padded_mask shape: {padded_mask.shape}")
        # get full mask
        mask_tensor = torch.stack(branch_mask, dim=0).cuda()
        b, l, s = mask_tensor.shape
        expanded_mask = mask_tensor.unsqueeze(1).repeat(1, 8, 1, 1).cuda()  # [B, num_heads, S, S]
        expanded_mask = expanded_mask.view(b * 8, s, s).cuda()  # [B * num_heads, S, S]

        # padding to [B, R, D]
        padded_feats = pad_sequence(branch_feats, batch_first=True, padding_value=0).cuda()

        # inter branch attention
        out, _ = self.inter_branch_attention(query=padded_feats, key=padded_feats, value=padded_feats,
                                             need_weights=False, attn_mask=expanded_mask)

        # average pooling
        out = torch.sum(out, dim=1)
        return out