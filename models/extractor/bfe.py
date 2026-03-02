import torch, dgl
import torch.nn as nn
import torch.nn.functional as F


class PathAdd(nn.Module):
    """
        Add - Based Path Processor, do not need to be trainable
        Readout Func
    """
    def __init__(self):
        super(PathAdd, self).__init__()

    def message_func(self, edges):
        return {
            "feats": edges.src["feats"]
        }

    def reduce_func(self, nodes):
        feats = nodes.mailbox["feats"]
        # notice: sum
        feats = feats.sum(-2)
        return {
            "feats": feats
        }

    def apply_node_func(self, nodes):
        return {
            "feats": nodes.data["feats"]
        }

class PathLstm(nn.Module):
    def __init__(self, feat_size, hidden_size):
        super(PathLstm, self).__init__()

        self.feat_size = feat_size
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=feat_size, hidden_size=hidden_size, batch_first=True)
        

    def message_func(self, edges):
        return {
            "feats": edges.src["feats"]
        }

    def reduce_func(self, nodes):
        feats = nodes.mailbox["feats"]
        # notice: sum
        # feats = feats.sum(-2)
        lstm_out, _ = self.lstm(feats)
        feats = lstm_out[:, -1, :]  # shape: [num_nodes, hidden_size]

        # print(f"lstm_out shape: {lstm_out.shape}")
        return {
            "feats": feats
        }

    def apply_node_func(self, nodes):
        return {"feats": nodes.data["feats"]}