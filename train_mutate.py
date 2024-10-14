import argparse
import datetime
import os
import random
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import HeteroData, Batch

from daos.mutate import YulASTMutator, mutate_operator, mutate_stat_del, MutationDataset
from daos.sol_mutate import SolASTMutator, AOR, BOR, UOR, PKR, FVR, EED, MOD, SolMutationDataset
from daos.transform import TransformSequence, format_data_type
from daos.yul_ast import YulASTDataset
from daos.sol_ast import SolASTDataset
from models.simloss import SimLoss
from models.unimp import UniMP
from settings import PROJECT_PATH

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from models.roberta import RoBERTa


class CLRModel(torch.nn.Module):
    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.doc_model = RoBERTa(device=device, **kwargs)
        self.graph_model = UniMP(**kwargs).to(device)
        self.criterion = SimLoss(embedding_dim=kwargs['out_channels']).to(device)

    def forward(self, batch_data: HeteroData):
        graph_feats = self.graph_model(batch_data)
        doc_feats = self.doc_model(batch_data.prompt)
        return self.criterion(doc_feats, graph_feats)


def train(
        data_path: str,
        model_args: dict,
        **kwargs,
) -> torch.nn.Module:
    device = torch.device('cpu')
    if kwargs.get('gpu'):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU for speedup...')
        else:
            print('Warning: GPU is not available, using CPU for training...')

    # build dataset
    dataset = SolASTDataset(
        root=data_path,
        currency=os.cpu_count() // 2,
        transform=TransformSequence([
            format_data_type,
            # mutate,
        ])
    )
    dataset = MutationDataset(
        dataset=dataset,
        mutator=SolASTMutator(
            mutate_funcs=[
                AOR,
                BOR,
                UOR,
                PKR,
                FVR,
                EED,
                MOD
            ],
            max_mutant=15,
        ),
    )
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=kwargs.get('batch_size', 16),
        num_workers=kwargs.get('num_workers', 0),
        collate_fn=lambda _batch: _batch,
    )

    # init models
    model = CLRModel(**{
        "metadata": dataset.metadata,
        **model_args,
    })
    if os.path.exists(kwargs['pretrain_path']):
        model.load_state_dict(
            state_dict=torch.load(kwargs['pretrain_path']) if torch.cuda.is_available()
            else torch.load(kwargs['pretrain_path'], map_location=torch.device('cpu')),
            strict=False,
        )
    graph_model = model.graph_model.to(device)
    for params in graph_model.convs.parameters():
        params.requires_grad = False
    for params in graph_model.bns.parameters():
        params.requires_grad = False
    for params in graph_model.node_lins.parameters():
        params.requires_grad = False
    for params in graph_model.edge_lins.parameters():
        params.requires_grad = False

    # define loss and optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=kwargs.get('lr', 1e-3),
        weight_decay=kwargs.get('weight_decay', 5e-4),
    )

    # start training
    print('start training...')
    model.train()
    report_step = kwargs.get('report_step', 5)
    batch_size = kwargs.get('batch_size', 32)
    for epoch in range(kwargs.get('epoch')):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        total_loss, batch_cnt, sample_cnt = 0, 0, 0

        for batch_data in dataloader:
            for i in range(batch_size):
                try:
                    mutants = batch_data[i]
                    if len(mutants) < 6:
                        continue

                    # mutants = Batch.from_data_list(mutants)
                    mutants = mutants.to(device)
                    pred = graph_model(mutants)
                    pred = pred / pred.norm(p=2, dim=-1, keepdim=True)
                    pred = pred[0] @ pred.T
                    pred = torch.softmax(pred.flatten(), dim=0)
                    target = [1] + [0 for _ in range(len(mutants) - 1)]
                    target = torch.tensor(target, dtype=torch.float, device=device)
                    loss = F.cross_entropy(pred, target) / len(mutants)
                    total_loss += loss.detach() / len(mutants)
                    loss.backward()
                except:
                    print('Warning: CUDA OOM at #{}'.format(sample_cnt))
                    traceback.print_exc()
                finally:
                    sample_cnt += 1

            # update params
            optimizer.step()
            optimizer.zero_grad()
            batch_cnt += 1
            if batch_cnt % report_step == 0:
                print('{}, epoch #{}, batch #{}, loss {}'.format(
                    datetime.datetime.now(), epoch,
                    batch_cnt, total_loss / report_step
                ))
                total_loss = 0
                if batch_cnt % 20 == 0:
                    save_path = os.path.join(PROJECT_PATH, 'model_mutate2.pth')
                    print('training process finished! save the model to: %s' % save_path)
                    torch.save(model.state_dict(), save_path)

    # detach to cpu and return
    model.graph_model = graph_model.to('cpu')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--tuning_layers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--report_step', type=int, default=5)
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()

    torch.random.manual_seed(43)
    model = train(
        data_path=args.data_path,
        model_args=dict(
            hidden_channels=args.hidden_channels,
            out_channels=args.hidden_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            tuning_layers=args.tuning_layers,
        ), **{
            'lr': args.lr,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'num_workers': args.num_workers,
            'report_step': args.report_step,
            'pretrain_path': args.pretrain_path,
            'gpu': args.gpu,
        }
    )

    save_path = os.path.join(PROJECT_PATH, 'model_mutate2.pth')
    print('training process finished! save the model to: %s' % save_path)
    torch.save(model.state_dict(), save_path)
