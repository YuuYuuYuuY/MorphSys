import torch, dgl
import torch.nn as nn
from dgl.nn.pytorch import GINConv, AvgPooling, SumPooling, MaxPooling
from models.extractor.ib import InterBranch
from torch.nn.functional import relu

class GinInterBranch(nn.Module):
    def __init__(self, x_size, h_size, num_classes, num_layers=2, fc=True, bn=False, pool="sumpool", path="add", direction="out", fuse="weighted"):
        super(GinInterBranch, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.num_layers = num_layers
        # gnn layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        if bn:
            for i in range(num_layers):
                if i:
                    mlp = nn.Sequential(nn.Linear(self.h_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                else:
                    mlp = nn.Sequential(nn.Linear(self.x_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                conv = GINConv(mlp, learn_eps=True, activation=relu)
                bn = nn.BatchNorm1d(h_size)
                self.convs.append(conv)
                self.batchnorm.append(bn)
        else:
            for i in range(num_layers):
                if i:
                    mlp = nn.Sequential(nn.Linear(self.h_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                else:
                    mlp = nn.Sequential(nn.Linear(self.x_size, self.h_size), nn.ReLU(), nn.Linear(self.h_size, self.h_size))
                conv = GINConv(mlp, learn_eps=True)
                self.convs.append(conv)
         
        if pool == "sumpool":
            self.pool = SumPooling()
        elif pool == "avgpool":
            self.pool = AvgPooling()
        elif pool == "maxpool":
            self.pool = MaxPooling()
        self.fc = fc
        # self.linear = nn.Linear(h_size, num_classes)  # 未使用 num_classes = 0

        # inter-branch attention
        self.inter_branch = InterBranch(embed_dim=self.h_size, num_heads=8, batch_first=True, path=path, direction=direction)
        self.r = nn.Parameter(torch.tensor(0.5))
        self.fuse = fuse
        print(f"self.fuse :{self.fuse}")

    def forward_backbone(self, batch):
        # a batch of graphs is put in to gpu for computation
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph (异源图)
        g = dgl.graph(g.edges())  # 只留下edges()
        n = g.number_of_nodes()
        # feed embedding
        feats = batch.feats.cuda()
        # print(f"before fwd, batch size is {batch.feats.shape}")
        for i in range(self.num_layers):
            # print(f"At this step, feats size is {feats.shape}")
            feats = self.convs[i](g, feats)
            feats = self.batchnorm[i](feats)
            # print(f"in fwd, feats size is {feats.shape}")
        # 在这一步上使用 SFA
        # feats = self.calculate(feats)

        temp_logits = self.pool(batch.graph.to(torch.device("cuda")), feats)

        g.ndata["feats"] = feats
        # propagate
        out = self.inter_branch(g, batch.max_leaf_len, batch.offset_leaf)

        # print(f"logits: {logits}")
        # print(f"out: {out}")
        # 返回B个图进行lstm操作后的 [B, D]  like readout function
        # print(f"logits shape is {logits.shape}")
        return temp_logits, out

    def forward(self, batch):
        logits, out = self.forward_backbone(batch)
        if self.fc:
            pass
            return logits
        else:
            # print(f"after forward logits size is {logits.shape}")
            # print(f"after forward out size is {out.shape}")
            # return logits
            if self.fuse == "weighted":
                return logits * self.r + out * (1-self.r)
            else:
                return torch.add(logits, out)
            # return torch.cat((logits, out), dim=1)


