import hashlib
import sys
sys.path.append("..")

import asyncio
import csv
import datetime
import json
import multiprocessing
import subprocess
import os
import re
import copy
from typing import List, Dict

import torch
from torch_geometric.data import Dataset, HeteroData

from compiler import compile_by_solcjs
from ir.yul.function import sol_extend_function, doc_ref
from ir.yul.feats import get_sol_feats
from daos.yul_ast import YulASTDataset


class SolASTDataset(YulASTDataset):
    def process(self):
        # check if data is available
        index_path = os.path.join(self.processed_dir, 'index.csv')
        metadata_path = os.path.join(self.processed_dir, 'metadata.json')
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            return

        # generate hetero data
        fut = self.process_async()
        asyncio.get_event_loop().run_until_complete(fut)

        # build index file
        if self.log:
            print('{} building index and metadata...'.format(datetime.datetime.now()))
        index = 0
        index_file = open(
            index_path, 'w',
            encoding='utf-8', newline='\n',
        )
        index_writer = csv.writer(index_file)
        index_writer.writerow(['index', 'filename', 'offset', 'doc_hash', 'contract', 'function', 'comment', 'code'])
        node_types, edge_types = set(), set()
        for fn in filter(
                lambda _fn: _fn.startswith('0x') and _fn.endswith('.pt'),
                os.listdir(self.processed_dir)
        ):
            path = os.path.join(self.processed_dir, fn)
            try:
                for offset, data in enumerate(torch.load(path)):
                    doc_hash = hashlib.sha256(data.documentation.encode()).hexdigest()
                    index_writer.writerow([index, fn, offset, doc_hash, data.contract, data.sign, data.documentation, data.code.replace('\n', '').replace('\r', '')])
                    node_types.update(data.node_types)
                    edge_types.update(data.edge_types)
                    index += 1
            except Exception as e:
                print(f"Error loading file {path}: {e}")

        index_file.close()
        with open(metadata_path, 'w') as f:
            json.dump([list(node_types), list(edge_types)], f)

    async def load_extended_asts(
            self, standard_json: dict,
            solc_version: str,
            **kwargs,
    ) -> List[Dict]:
        product = await compile_by_solcjs(
            standard_json=standard_json,
            solc_version=solc_version,
        )
        # 打印输出结果

        if product is None:
            return []
        # fetch all sol ast func nodes using bfs
        func_asts = list()
        for source in product['sources'].values():
            q = [(None, source['ast'])]
            while len(q) > 0:
                contract, node = q.pop(0)
                if not isinstance(node, dict) or not node.get('nodeType'):
                    continue
                if node['nodeType'] == 'FunctionDefinition':
                    node['contract'] = contract
                    func_asts.append(node)
                    continue

                # bfs expanding
                if node['nodeType'] == 'ContractDefinition':
                    contract = node['name']
                for val in node.values():
                    if isinstance(val, dict):
                        q.append((contract, val))
                    if isinstance(val, list):
                        val = [(contract, v) for v in val]
                        q.extend(val)
      # attach source code to func nodes
        fid2filename = {
            source['id']: fn
            for fn, source in product['sources'].items()
        }
        id2func_node = {
            n.get('id') : n for n in func_asts
        }
        extended_funcs = []
        for func in func_asts:
            begin, length, fid = [int(n) for n in func['src'].split(':')]
            filename = fid2filename[fid]
            code = standard_json['sources'][filename]['content']
            func['code'] = code[begin: begin + length]
            func['address'] = kwargs['contract_address']
            # add doc for func node
            func['call'] = []
            '''
            compiled_out = product['contracts'][filename][func['contract']]
            docs = compiled_out['devdoc']['methods']
            docs.update(compiled_out['userdoc']['methods'])
            for func_sign in docs.keys():
                rlt = re.search('%s(.*?)' % func['name'], func_sign)
                if rlt is None:
                    continue
                func['documentation'] = str(docs[func_sign])
                break
            '''
            doc = func.get('documentation')
            comment = doc if isinstance(doc, str) else doc.get('text', '') if isinstance(doc, dict) else ''
            if comment:   
                comment = re.sub(r'[\r\n]+', ' ', comment)
                func['doc'] = doc_ref(comment)
  

        for func in func_asts:
            if not func.get('doc'):
                continue 
            # extend node
            func = copy.deepcopy(func)
            func = sol_extend_function(func, id2func_node)
            extended_funcs.append(func)
        return extended_funcs

    @staticmethod
    def save_ast2file(
            lock: multiprocessing.Semaphore,
            asts: List[Dict], path: str
    ):
        torch.save([SolASTDataset.ast2data(ast) for ast in asts], path)
        lock.release()

    @staticmethod
    def ast2data(ast: dict) -> HeteroData:
        q = [(0, ast)]
        ast_obj2index = {id(ast): 0}
        feat = get_sol_feats(ast)

        node_type2feats = {ast['nodeType']: [[
            0,
            *[int(n) for n in ast['src'].split(':')],
        ] + feat ] }
        
        node_type2srcs = {ast['nodeType']: [[
            int(n) for n in ast['src'].split(':')]
        ]}
        edge_type2edges = dict()
        edge_type2attrs = dict()
        while len(q) > 0:
            depth, node = q.pop(0)

            # scan sub nodes
            sub_nodes = list()
            for k, v in node.items():
                if isinstance(v, dict):
                    sub_nodes.append((k, 0, v))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        sub_nodes.append((k, i, item))

            # build sub node index and edges
            for rela, order, sub_node in sub_nodes:
                if not isinstance(sub_node, dict) or not sub_node.get('nodeType'):
                    continue
                q.append((depth + 1, sub_node))
                if not node_type2feats.get(sub_node['nodeType']):
                    node_type2feats[sub_node['nodeType']] = list()
                    node_type2srcs[sub_node['nodeType']] = list()

                # build node feats
                idx = len(node_type2feats[sub_node['nodeType']])
            
                feats = [
                    depth + 1,
                    *[int(n) for n in sub_node['src'].split(':')],
                ]
                # get feat
                feat = get_sol_feats(sub_node)
                if feat != None:
                    feats.extend(feat)
                # add 
                node_type2feats[sub_node['nodeType']].append(feats)
                node_type2srcs[sub_node['nodeType']].append([
                    int(n) for n in sub_node['src'].split(':')
                ])
                ast_obj2index[id(sub_node)] = idx

                # add sub edges
                edge_type = node['nodeType'], rela, sub_node['nodeType']
                if not edge_type2edges.get(edge_type):
                    edge_type2edges[edge_type] = list()
                    edge_type2attrs[edge_type] = list()
                edge_type2edges[edge_type].append([
                    ast_obj2index[id(node)],
                    ast_obj2index[id(sub_node)],
                ])
                edge_type2attrs[edge_type].append([order])

                # add rev-sub edges
                edge_type = sub_node['nodeType'], '_%s' % rela, node['nodeType']
                if not edge_type2edges.get(edge_type):
                    edge_type2edges[edge_type] = list()
                    edge_type2attrs[edge_type] = list()
                edge_type2edges[edge_type].append([
                    ast_obj2index[id(sub_node)],
                    ast_obj2index[id(node)],
                ])
                edge_type2attrs[edge_type].append([order])

        # format hetero data
        data = HeteroData()
        data['code'] = ast.get('code','')
        data['contract'] = ast.get('contract','')
        data['address'] = ast.get('address','')
        data['sign'] = ast.get('name','')
        data['documentation'] = ast.get('doc','')
        data['call'] = ast.get('call')
        for node_type, feats in node_type2feats.items():
            data[node_type].x = torch.tensor(feats)
        for node_type, srcs in node_type2srcs.items():
            data[node_type].src = torch.tensor(srcs)
        for edge_type, edge_index in edge_type2edges.items():
            data[edge_type].edge_index = torch.tensor(edge_index).t().contiguous()
        for edge_type, edge_attr in edge_type2attrs.items():
            data[edge_type].edge_attr = torch.tensor(edge_attr)
        return data


if __name__ == '__main__':
    import traceback
    import sol_mutate
    from daos.transform import TransformSequence, format_data_type, format_prompt, sol_format_prompt
    path = r'CCIHunter/UniIntent/SmartCoCo'
    # curr = os.cpu_count() // 2
    dataset = SolASTDataset(
        root=path, currency=1,
        transform=TransformSequence([
            format_data_type,
            sol_format_prompt,
        ])
    )


