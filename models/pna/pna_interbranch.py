import torch, dgl
import torch.nn as nn
from dgl.nn.pytorch import PNAConv, SumPooling, AvgPooling, MaxPooling
from models.extractor.ib import InterBranch

class PnaInterBranch(nn.Module):
    def __init__(self, x_size, h_size, num_classes, num_layers=2, fc=True, bn=False, pool="sumpool", path="add", direction="out", fuse="weighted"):
        super(PnaInterBranch, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.num_layers = num_layers
        # gnn layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        # mlp 映射到128维空间
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )

        if bn:
            for i in range(num_layers):
                conv = PNAConv(in_size=h_size, out_size=h_size, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification'], delta=2.5)
                bn = nn.BatchNorm1d(h_size)
                self.convs.append(conv)
                self.batchnorm.append(bn)
        else:
            for i in range(num_layers):
                conv = PNAConv(in_size=h_size, out_size=h_size, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification'], delta=2.5)
                self.convs.append(conv)
        # pooling layer as readout function
         
        if pool == "sumpool":
            self.pool = SumPooling()
        elif pool == "avgpool":
            self.pool = AvgPooling()
        elif pool == "maxpool":
            self.pool = MaxPooling()

        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)
        self.inter_branch = InterBranch(embed_dim=self.h_size, num_heads=8, batch_first=True, path=path, direction=direction)
        self.r = nn.Parameter(torch.tensor(0.5))
        self.fuse = fuse
        

    def forward_backbone(self, batch):
        # a batch of graphs is put in to gpu for computation
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph (异源图)
        # 只留下edges()目的：只需要边的连接结构和节点嵌入（feats），而不需要原始图的节点或边属性这一步可以确保只保留连接信息，从而避免复杂的图结构导致的错误或不匹配
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        feats = self.mlp1(batch.feats.cuda())
        feats_copy = feats.clone()
        # print(f"At this step, batch size is {feats.shape}")
        res = feats
        for i in range(self.num_layers):
            # print(f"At this step, feats size is {feats.shape}")
            res = self.convs[i](g, res, feats)
            # print(f"After this step, feats size is {feats.shape}")
            res = self.batchnorm[i](res)
            # print(f"in fwd, feats size is {feats.shape}")
        # 在这一步上使用 SFA
        # feats = self.calculate(feats)
        # print(f"At this step, batch size is {feats.shape}")
        # shape eror
        logits = self.pool(batch.graph.to(torch.device("cuda")), res)
        # 返回B个图进行lstm操作后的 [B, D]  like readout function
        g.ndata["feats"] = feats_copy

        # propagate
        out = self.inter_branch(g, batch.max_leaf_len, batch.offset_leaf)
        return logits, out

    def forward(self, batch):
        logits, out = self.forward_backbone(batch)
        # print(f"after forward logits size is {logits.shape}")
        # print(logits)
        if self.fc:
            pass
            return logits
        else:
            if self.fuse == "weighted":
                return logits * self.r + out * (1-self.r)
            else:
                return torch.add(logits, out)
            # return torch.cat((logits, out), dim=1)


