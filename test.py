import argparse
import asyncio
import json
import os
import re
from typing import Dict

import joblib
import torch
from torch_geometric.data import HeteroData

from daos.downloader import ContractSourceDownloader
from daos.transform import format_data_type, sol_format_prompt
from daos.sol_ast import SolASTDataset
from settings import MISC_PATH
from train_tuning import CLRModel
from utils.standard_json import get_sol_standard_json


def get_emb(model: CLRModel, data):
    '''
    Get text data or heterogeneous data embedding
    :param model: Model used to obtain embeddings
    :param data: Text data or heterogeneous data
    :return: Embedding of data
    '''
    if isinstance(data, str):
        doc_feat = model.doc_model(data)
        doc_embed = model.criterion.text_feats_lin(doc_feat)
        return doc_embed / doc_embed.norm(p=2, dim=-1, keepdim=True)
    else:
        graph_feat = model.graph_model(data)
        graph_embed = model.criterion.graph_feats_lin(graph_feat)
        return graph_embed / graph_embed.norm(p=2, dim=-1, keepdim=True)


def match_calculate(
        model: CLRModel,
        data: HeteroData,
        rf_path: str, **kwargs
) -> bool:
    '''
    Calculating the match score between code and comments
    :param model: Model used to obtain embeddings
    :param data: HeteroData of a function
    :return: Match scores between the function name, and the code and comments of the function,
             with higher scores indicating greater consistency
    '''

    rf_model = joblib.load(rf_path)
    device = torch.device('cuda' if kwargs['gpu'] else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        feature = model(data)
        feature = feature.cpu().detach().numpy()
        result = rf_model.predict(feature)
    return result[0].item() == 0


def sim_calculate(model: CLRModel, data1: HeteroData, data2: HeteroData):
    '''
    Calculate the code logic similarity score for the two input functions
    :param model: Model used to obtain embeddings
    :param data1: HeteroData of a function1
    :param data2: HeteroData of a function2
    :return: function name, and the code logic similarity score of the two functions entered,
             the higher the score, the closer the logic is
    '''
    graph_embed1 = get_emb(model, data1)
    graph_embed2 = get_emb(model, data2)
    score = (graph_embed1 @ graph_embed2.T) * 100
    return data1.contract + "-" + data1.sign, score[0].item()


async def get_json(contract_address: str):
    '''
    Crawling contract json with the help of api
    :param contract_address: Address of the target contract
    :return: Contract json
    '''
    d = ContractSourceDownloader('https://api.etherscan.io/api?apikey=?')
    result = await d.download(
        contract_address=contract_address,
    )
    return result


def json2data(data, filename):
    '''
    from json_data to heterodata
    :param data: json_data from etherscan or local file
    :return: heterodata of the contract
    '''
    # if data is from hardhat
    if 'input' in data:
        standard_json = data['input']
        solc_version = data['solcVersion']
        standard_json['settings']['outputSelection']={
                    "*": {
                        "*": ['userdoc', 'devdoc'],
                        "": ["ast"],
                    },
                }
    else:
        solc_version = re.search('v(.*?)\+commit', data["CompilerVersion"])
        solc_version = solc_version.group(1)
        # build standard json
        standard_json = get_sol_standard_json(
            code=data['SourceCode'][1:-1] if
            data['SourceCode'][0] == '{' and
            data['SourceCode'][-1] == '}' else
            data['SourceCode'],
            optimized=data["OptimizationUsed"] == '1',
            optimization_runs=int(data["Runs"]),
            
            output_selection={
                "*": {
                    "*": ['userdoc', 'devdoc'],
                    "": ["ast"],
                },
            }
        )
    
    task = SolASTDataset.load_extended_asts(None, standard_json, solc_version, **{
            'contract_address': filename.split('.') [0]
        })
    fun_asts = asyncio.run(task)
    fun_data = []
    node_types, edge_types = set(), set()
    for fun_ast in fun_asts:
        hetero_data = SolASTDataset.ast2data(fun_ast)
        hetero_data = format_data_type(hetero_data)
        hetero_data = sol_format_prompt(hetero_data)
        node_types.update(hetero_data.node_types)
        edge_types.update(hetero_data.edge_types)
        fun_data.append(hetero_data)
    metadata = [list(node_types), list(edge_types)]
    return fun_data, metadata


def load_model(metadata, model_args: dict, **kwargs):
    '''
    Loading models
    :param data: Used to provide metadata
    :param model_args: Model parameters
    :param kwargs: Model loading paths, etc.
    :return: Available CLRModel
    '''
    device = torch.device('cuda' if kwargs['gpu'] else 'cpu')
    model = CLRModel(**{
        "device": device,
        "metadata": metadata,
        **model_args,
    })
    model.load_state_dict(
        state_dict=torch.load(kwargs['pretrain_path']) if torch.cuda.is_available()
        else torch.load(kwargs['pretrain_path'], map_location=torch.device('cpu')),
        strict=False,
    )
    return model



def address2data(address):
    '''
    Implementing conversion from contract address to heterodata
    :param address: Address of the target contract
    :return: The ast of functions in a contract and metadata
    '''
    result = asyncio.run(get_json(address))
    data = json.loads(json.dumps(result))
    return json2data(data)



def localfile2data(filename):
    '''
    Implementing conversion from local json file to heterodata
    :param filename: Filename of the target contract
    :return: The ast of functions in a contract and metadata
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        result = json.load(f)
    data = json.loads(json.dumps(result))
    return json2data(data, filename)


def infer(file_name, rf_path, model_args: dict, **kwargs) -> Dict:
    '''
    Infer the code and comments of each function is consistent or not
    :param file_name: File_name of the target contract
    :param model_args: Model parameters
    :param kwargs: Model loading paths, etc.
    :return: A list containing the match scores between the code and comments for each function
    '''
    fun_data, metadata = localfile2data(file_name)
    model = load_model(metadata, model_args, **kwargs)

    rlt = dict()
    for data in fun_data:
        if kwargs['gpu']:
            data = data.to('cuda')
        func_name = '%s-%s' % (data.contract, data.sign)
        is_consistent = match_calculate(model, data, rf_path, **kwargs)
        rlt[func_name] = is_consistent
    return rlt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--rf_path', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--tuning_layers', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()
    model_args = dict(
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tuning_layers=args.tuning_layers,
    )

    rlt = infer(
        args.data_path,
        args.rf_path,
        model_args, **{
            'pretrain_path': args.pretrain_path,
            'gpu': args.gpu,
        }
    )
    rlt = json.dumps(rlt, indent=4)
    print(rlt)