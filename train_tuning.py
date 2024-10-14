import numpy as np
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import csv
import datetime
import functools
import os
import traceback
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
import joblib
from sklearn.metrics import classification_report
from torch.utils.data import random_split, RandomSampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from daos.transform import TransformSequence, fail_label_binding, format_data_type, label_binding, format_prompt, sol_format_prompt
from daos.yul_ast import YulASTDataset
from daos.sol_ast import SolASTDataset
from models.simloss import SimLoss
from models.unimp import UniMP
from models.roberta import RoBERTa
from settings import PROJECT_PATH
from utils.sampler import DeduplicateBatchSampler
from torch.utils.data.sampler import BatchSampler
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class CLRModel(torch.nn.Module):
    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.doc_model = RoBERTa(device=device, **kwargs)
        self.graph_model = UniMP(**kwargs).to(device)
        self.criterion = SimLoss(embedding_dim=kwargs['out_channels']).to(device)

    def forward(self, batch_data: HeteroData):
        graph_embeds = self.graph_model(batch_data)
        graph_embeds_nor = graph_embeds / graph_embeds.norm(p=2, dim=-1, keepdim=True)
        docs_embeds = self.doc_model(batch_data.prompt)
        docs_embeds_nor = docs_embeds / docs_embeds.norm(p=2, dim=-1, keepdim=True)
        pred1 = docs_embeds_nor @ graph_embeds_nor.T 
        pred1 = torch.diagonal(pred1).unsqueeze(1) / self.criterion.temperature.exp() 
        return torch.cat((docs_embeds, graph_embeds, pred1), dim=-1)
        

def load_labels(label_path: str) -> Dict:
    rlt = dict()
    with open(label_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = '{}@{}@{}'.format(row['address'].split('.')[0], row['contract'], row['function'])
            rlt[key] = False if row['label'] == '1' else True
    return rlt


def fail_labels(label_path: str) -> Dict:
    rlt = dict()
    with open(label_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = '{}@{}@{}'.format(row['address'].split('.')[0], row['contract'], row['function'])
            rlt[key] = True if row['label'] == '0' else False
    return rlt

def train(
        data_path: str,
        model_args: dict,
        **kwargs,
):
    # device
    device = torch.device('cpu')
    if kwargs.get('gpu'):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU for speedup...')
        else:
            print('Warning: GPU is not available, using CPU for training...')

    # init dataset
    label = load_labels(os.path.join(data_path, 'manually labeled.csv'))

    dataset = SolASTDataset(
        root=data_path,
        currency=os.cpu_count()//2,
        transform=TransformSequence([
            format_data_type,
            sol_format_prompt,
            functools.partial(label_binding, label),
        ])
    )
    '''
    success_dataset = []
    fail_dataset = []
    cnt = 0
    for data in dataset:
        if data['fail_label']:
            fail_dataset.append(data)
            cnt += 1
            print(cnt)
        else:
            success_dataset.append(data)
    '''

    # init model
    model = CLRModel(**{
        "metadata": dataset.metadata,
        "device": device,
        **model_args,
    })
    if os.path.exists(kwargs['pretrain_path']):
        model.load_state_dict(
            state_dict=torch.load(kwargs['pretrain_path']) if torch.cuda.is_available()
            else torch.load(kwargs['pretrain_path'], map_location=torch.device('cpu')),
            strict=False,
        )

    torch.manual_seed(36)
    train_dataset, test_dataset = random_split(dataset, [kwargs['ratio'], 1- kwargs['ratio']])
    '''
    train_dataset, test = random_split(success_dataset, [kwargs['ratio'], 1- kwargs['ratio']])
    test_dataset = fail_dataset
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("dataset OK")


    # start training
    print('start training...')
    train_dataset = train_dataset.dataset
    dsampler = BatchSampler(
        sampler=RandomSampler(train_dataset),
        batch_size=kwargs.get('batch_size', 32),
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=dsampler,
        num_workers=kwargs.get('num_workers', 0),
    )

    model.eval()

    models_rf = [
        {'name': 'rf_model2', 'params': {'n_estimators': 20, 'class_weight': 'balanced', 'max_features': 'sqrt', 'bootstrap': False}}
    ]

    rf_models = []
    
    for model_param in models_rf:
        rf_models.append(RandomForestClassifier(**model_param['params']))

    x_train_all = []
    y_train_all = []
    for batch_data in tqdm(dataloader, total=len(dataloader), desc='Modeling'):
        try:
            with torch.no_grad():
                # x_train1 -> gpu -> cpu
                batch_data = batch_data.to(device)
                x_train = model(batch_data)
                y_train = batch_data.label.int()
                x_train_all.append(x_train.cpu().detach().numpy())
                y_train_all.append(y_train.cpu().detach().numpy())
        except torch.cuda.OutOfMemoryError:
            print('Warning: CUDA OOM')
            traceback.print_exc()
    x_train_all = np.concatenate(x_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)

    print("strat fitting...")
    for rf_model in tqdm(rf_models, total=len(rf_models), desc='Fitting'):
        rf_model.fit(x_train_all, y_train_all)
        save_path = os.path.join(PROJECT_PATH, f'rf_model_xiaorong.pth')
        joblib.dump(rf_model, save_path)

    print('evaluating on testing set...')
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=kwargs.get('batch_size', 16),
        num_workers=kwargs.get('num_workers', 10),
    )
    
    x_test_all = []
    y_test_all = []
    for batch_data in tqdm(dataloader, total=len(dataloader), desc='Testing'):
        try:
            with torch.no_grad():
                batch_data = batch_data.to(device)
                x_test = model(batch_data)
                y_test = batch_data.label.int()
                x_test_all.append(x_test.cpu().detach().numpy())
                y_test_all.append(y_test.cpu().detach().numpy())
        except torch.cuda.OutOfMemoryError:
            print('Warning: CUDA OOM')
            traceback.print_exc()
    x_test_all = np.concatenate(x_test_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)

    all_preds = []
    for i, rf_model in enumerate(rf_models):
        print("model",i+1)
        preds_i = rf_model.predict(x_test_all)
        print(classification_report(y_test_all, preds_i))
        all_preds.append(preds_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--tuning_layers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--report_step', type=int, default=5)
    parser.add_argument('--gpu', type=bool, default=False)

    args = parser.parse_args()

    torch.random.manual_seed(71)

    train(
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
            'ratio': 0.8
        }
    )

