import argparse
import datetime
import os
import sys
import traceback

import torch
from torch.utils.data import RandomSampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from daos.sol_ast import SolASTDataset
from daos.yul_ast import YulASTDataset
from models.simloss import SimLoss
from models.unimp import UniMP
from settings import PROJECT_PATH
from daos.transform import TransformSequence, format_prompt, format_data_type, sol_format_prompt

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from models.roberta import RoBERTa
from utils.sampler import DeduplicateBatchSampler


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
    print(sys.prefix)
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
            sol_format_prompt,
        ])
    )

    dsampler = DeduplicateBatchSampler(
        sampler=RandomSampler(dataset[610000: -1]),
        batch_size=kwargs.get('batch_size', 16),
        drop_last=True,
    )
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=dsampler,
        num_workers=kwargs.get('num_workers', 0),
    )
    print("dataset OK")
    # init models
    model = CLRModel(**{
        "device": device,
        "metadata": dataset.metadata,
        **model_args,
    })
    if os.path.exists(kwargs['pretrain_path']):
        model.load_state_dict(
            state_dict=torch.load(kwargs['pretrain_path']) if torch.cuda.is_available()
            else torch.load(kwargs['pretrain_path'], map_location=torch.device('cpu')),
            strict=False,
        )
    print("load pretrain model OK!")
    # define loss and optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=kwargs.get('lr', 1e-3),
        weight_decay=kwargs.get('weight_decay', 5e-4),
    )

    print(len(dataset))
    # start training
    print('start training...')
    model.train()
    report_step = kwargs.get('report_step', 5)
    for epoch in range(kwargs.get('epoch')):
        total_loss, batch_cnt = 0, 0
        for batch_data in dataloader:
            optimizer.zero_grad()
            try:
                #print("-------start-------",datetime.datetime.now())
                batch_data = batch_data.to(device)
                #print("----data2device----",datetime.datetime.now())
                loss = model(batch_data)
                #print("------getloss------",datetime.datetime.now())
                total_loss += loss.detach()
                #print("------addloss------",datetime.datetime.now())
                loss.backward()
                #print("-----backward------",datetime.datetime.now())
                optimizer.step()
                #print("--------step-------",datetime.datetime.now())
            except:
                print('Warning: CUDA OOM at batch #{}'.format(batch_cnt))
                traceback.print_exc()

            batch_cnt += 1
            if batch_cnt % report_step == 0:
                print('{}, epoch #{}, batch #{}, loss {}'.format(
                    datetime.datetime.now(), epoch,
                    batch_cnt, total_loss / report_step
                ))
                total_loss = 0

            if batch_cnt % 20 == 0:
                save_path = os.path.join(PROJECT_PATH, 'model.pth')
                print('Save the model to: %s' % save_path)
                torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    print("start!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--tuning_layers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--report_step', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default='')
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
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'report_step': args.report_step,
            'pretrain_path': args.pretrain_path,
            'gpu': args.gpu,
        }
    )

    save_path = os.path.join(PROJECT_PATH, 'model.pth')
    print('training process finished! save the model to: %s' % save_path)
    torch.save(model.state_dict(), save_path)
