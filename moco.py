import torch
import torch.nn as nn

# import models
from models.treelstm import TreeLSTM, TreeLSTMv2, TreeLSTMDouble
from models.gin_lstm_mlp import GinLstmMlp

# pool models
from models.gin.gin_pool import GinPool
from models.gat.gat_pool import GatPool
from models.agnn.agnn_pool import AgnnPool
from models.appnp.appnp_pool import AppnpPool
from models.dgn.dgn_pool import DgnPool
from models.dotgat.dotgat_pool import DotgatPool
from models.gatedgraph.gatedgraph_pool import GatedGraphPool
from models.gcn2.gcn2_pool import Gcn2Pool
from models.pna.pna_pool import PnaPool
from models.sgc.sgc_pool import SgcPool
from models.tag.tag_pool import TagPool
from models.twirls.twirls_pool import TwirlsPool
from models.sage.sage_pool import SagePool

# with interbranch
from models.gin.gin_interbranch import GinInterBranch
from models.gat.gat_interbranch import GatInterBranch
from models.agnn.agnn_interbranch import AgnnInterBranch
from models.appnp.appnp_interbranch import AppnpInterBranch
from models.dgn.dgn_interbranch import DgnInterBranch
from models.dotgat.dotgat_interbranch import DotgatInterBranch
from models.gatedgraph.gatedgraph_interbranch import GatedGraphInterBranch 
from models.gcn2.gcn2_interbranch import Gcn2InterBranch
from models.pna.pna_interbranch import PnaInterBranch
from models.sgc.sgc_interbranch import SgcInterBranch
from models.tag.tag_interbranch import TagInterBranch
from models.twirls.twirls_interbranch import TwirlsInterBranch
from models.sage.sage_interbranch import SageInterBranch


# used for model encoder selection
model_dic = {
    "treelstm": TreeLSTMDouble,
    "gin_lstm_mlp": GinLstmMlp,
    
    # pool models
    "gin_pool": GinPool,
    "gat_pool": GatPool,
    "agnn_pool": AgnnPool,
    "appnp_pool": AppnpPool,
    "dgn_pool": DgnPool,
    "dotgat_pool": DotgatPool,
    "gatedGraph_pool": GatedGraphPool,
    "gcn2_pool": Gcn2Pool,
    "pna_pool": PnaPool,
    "sgc_pool": SgcPool,
    "tag_pool": TagPool,
    "twirls_pool": TwirlsPool,
    "sage_pool": SagePool,

    # with branch feature 
    "gin_interbranch": GinInterBranch,
    "gat_interbranch": GatInterBranch,
    "agnn_interbranch": AgnnInterBranch,
    "appnp_interbranch": AppnpInterBranch,
    "dgn_interbranch": DgnInterBranch,
    "dotgat_interbranch": DotgatInterBranch,
    "gatedGraph_interbranch": GatedGraphInterBranch,
    "gcn2_interbranch": Gcn2InterBranch,
    "pna_interbranch": PnaInterBranch,
    "sgc_interbranch": SgcInterBranch,
    "tag_interbranch": TagInterBranch,
    "twirls_interbranch": TwirlsInterBranch,
    "sage_interbranch": SageInterBranch,
}


class CustomizedMoCo(nn.Module):
    """
    self-defined class
    """
    def __init__(self, args, dim=128, K=256, m=0.99, T=0.1, symmetric=True):
        super(CustomizedMoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        if args.encoder == "treelstm":
            model_key = "treelstm"
        else:
            model_key = f"{args.encoder}_pool"
           
        if args.interbranch == 1:
            model_key = f"{args.encoder}_interbranch"

        model = model_dic[model_key]

        # create the encoders
        self.backbone_q = model(
            x_size=len(args.input_features),  # [2,3,4,12,13]
            h_size=args.h_size,  # 128
            num_classes=0,  # fc unused
            fc=False,
            bn=args.bn,
            pool=args.readout,
            path=args.path, 
            direction=args.direction,
            fuse=args.fuse
        )  # query encoder

        self.backbone_k = model(
            x_size=len(args.input_features),
            h_size=args.h_size,
            num_classes=0,  # fc unused
            fc=False,
            bn=args.bn,
            pool=args.readout,
            path=args.path, 
            direction=args.direction,
            fuse=args.fuse
        )  # key encoder

        dim_mlp = args.h_size

        # non-linear projection head, default setting

        # print(f"dim_projector: {dim_projector}")

        if args.interbranch == 0:
            if args.projector_bn:
                self.projector_q = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.BatchNorm1d(dim_mlp, affine=False),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, dim),
                )
                self.projector_k = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.BatchNorm1d(dim_mlp, affine=False),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, dim),
                )
            else:
                self.projector_q = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
                )
                self.projector_k = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
                )
        else:
            # with interbranch
            if args.projector_bn:
                self.projector_q = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.BatchNorm1d(dim_mlp, affine=False),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, dim),
                )
                self.projector_k = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.BatchNorm1d(dim_mlp, affine=False),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, dim),
                )
            else:
                self.projector_q = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
                )
                self.projector_k = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
                )
            
        self.encoder_q = nn.Sequential(self.backbone_q, self.projector_q)
        self.encoder_k = nn.Sequential(self.backbone_k, self.projector_k)
    


        # zip 将多个可迭代对象中的对应位置的元素打包成一个元组，然后返回由这些元组所组成的迭代器
        # 在此处即一个q 对应一个k
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # for param_q, param_k in zip(
        #     self.encoder_q.parameters(), self.encoder_k.parameters()
        # ):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.K % keys.shape[0] != 0:
            # K==256, 通常 keys.shape==128
            # should be sufficient to handle boundary cases.
            print("before", keys.shape)
            keys = torch.cat([keys, keys[-1:]], dim=0)
            print("after", keys.shape)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, view1, view2):
        # print("start")
        # compute query features, view1 stands for query, view2 stands for key
        q = self.encoder_q(view1)  # queries: NxC
        # print(f"q size: {q.shape}")
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(view2)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

        # print(f"k size: {k.shape}")
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # q, k 为同一batch增强生成的两个view故为正样本
        # negative logits: NxK 这里的K是queue size, .detach() 去掉梯度信息
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)  即 similarity matrix
        logits = torch.cat([l_pos, l_neg], dim=1)
        # print(f"In loss computation: logits shape is {logits.shape}")

        # apply temperature
        logits /= self.T

        # labels: positive key indicators 类别0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # print(f"In loss computation: labels shape is {labels.shape}")

        # InfoNCE
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, batch):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder, key encoder is updated by momentum
        # print("batch on : ", batch.device)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        view1, view2 = batch.view1, batch.view2
        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(view1, view2)
            loss_21, q2, k1 = self.contrastive_loss(view2, view1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss, default
            loss, q, k = self.contrastive_loss(view1, view2)

        self._dequeue_and_enqueue(k)

        return loss
