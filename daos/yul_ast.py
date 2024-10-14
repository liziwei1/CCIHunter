import sys
sys.path.append("..")
import asyncio
import copy
import csv
import datetime
import functools
import hashlib
import json
import multiprocessing
import os
import re
import sys
from typing import Generator, Dict, List, Union, Tuple, Callable

import torch
from torch_geometric.data import Dataset, HeteroData

from compiler import compile_by_solcjs
from compiler.solc import compile_yul_ast_by_solc
from ir.yul.function import extend_function
from ir.yul.scan import bfs_ast_node
from utils.opcode import get_evm_opcodes
from utils.standard_json import get_sol_standard_json


@functools.lru_cache(maxsize=2048)
def _load_data_list(_fn: str) -> List:
    return torch.load(_fn)


class YulASTDataset(Dataset):
    def __init__(
            self,
            root: str, currency: int = 3,
            timeout: int = 80,
            transform: Callable = None,
    ):
        self._currency = currency
        self._aio_lock = asyncio.Semaphore(self._currency)
        self._mp_lock = multiprocessing.Semaphore(self._currency)
        self.timeout = timeout
        self.log = True
        super().__init__(root, transform=transform)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['index.csv', 'metadata.json']

    @property
    def data_index(self) -> List:
        if getattr(self, '_data_index', None):
            return self._data_index

        rlt = list()
        path = os.path.join(self.processed_dir, 'index.csv')
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rlt.append({
                    'filename': row['filename'],
                    'offset': int(row['offset']),
                    'doc_hash': row['doc_hash'],
                })
        self._data_index = rlt
        return rlt

    @property
    def metadata(self):
        path = os.path.join(self.processed_dir, 'metadata.json')
        with open(path, 'r') as f:
            metadata = json.load(f)
        metadata[1] = [tuple(item) for item in metadata[1]]
        return metadata

    def len(self) -> int:
        return len(self.data_index)

    def get(self, idx: int) -> HeteroData:
        info = self.data_index[idx]
        path = os.path.join(self.processed_dir, info['filename'])
        data_list = _load_data_list(path)
        data = data_list[info['offset']]
        data['filename'] = info['filename'].split('.')[0]
        return data

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
        index_writer.writerow(['index', 'filename', 'offset', 'doc_hash', 'contract', 'sign'])
        node_types, edge_types = set(), set()
        for fn in filter(
                lambda _fn: _fn.endswith('.pt'),
                os.listdir(self.processed_dir)
        ):
            path = os.path.join(self.processed_dir, fn)
            for offset, data in enumerate(torch.load(path)):
                doc_hash = hashlib.sha256(data.documentation.encode()).hexdigest()
                index_writer.writerow([index, fn, offset, doc_hash, data.contract, data.sign])
                node_types.update(data.node_types)
                edge_types.update(data.edge_types)
                index += 1
        index_file.close()
        with open(metadata_path, 'w') as f:
            json.dump([list(node_types), list(edge_types)], f)

    async def process_async(self):
        index = 0
        for info in self.load_sources(self.raw_dir):
            index += 1

            # skip cache if available
            path = os.path.join(self.processed_dir, '%s.pt' % info['contract_address'])
            if os.path.exists(path):
                continue

            # logging
            if self.log:
                print('{} processing #{}: {}@{}, with solcv{}'.format(
                    datetime.datetime.now(), index,
                    info['contract_name'],
                    info['contract_address'],
                    info['solc_version'],
                ))

            # compile async
            await self._aio_lock.acquire()
            task = self.load_extended_asts(**info)
            task = asyncio.wait_for(task, timeout=self.timeout)
            task = asyncio.get_event_loop().create_task(task)
            task.add_done_callback(functools.partial(
                self.process_callback, info,
            ))

    def process_callback(self, info: dict, fut: asyncio.Future):
        asts = []
        try:
            asts = fut.result()
        except asyncio.TimeoutError:
            print('{} compile timeout: {}@{}, with solcv{}'.format(
                datetime.datetime.now(),
                info['contract_name'],
                info['contract_address'],
                info['solc_version'],
            ))
        finally:
            func_save = functools.partial(
                self.save_ast2file,
                self._mp_lock, asts,
            )

        # start data modeling processes
        self._mp_lock.acquire()
        self._aio_lock.release()
        p = multiprocessing.Process(
            target=func_save,
            args=(os.path.join(
                self.processed_dir,
                '{}.pt'.format(info['contract_address']),
            ),),
        )
        p.start()

    def load_sources(self, path: str) -> Generator:
        for fn in os.listdir(path):
            contract_address = fn.split('.')[0]
            fn = os.path.join(path, fn)
            with open(fn, 'r') as f:
                data = json.load(f)

            # format compiler version
            solc_version = re.search('v(.*?)\+commit', data["CompilerVersion"])
            if solc_version is None:
                continue
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

            # generate source project info
            yield {
                'contract_address': contract_address,
                'contract_name': data['ContractName'],
                'solc_version': solc_version,
                'standard_json': standard_json,
            }

    @staticmethod
    async def load_extended_asts(
            standard_json: dict,
            solc_version: str,
            **kwargs,
    ) -> List[Dict]:
        """
        Compile the standard json to get the ast of all functions of the contract
        :param standard_json: Standard json that can be used as input to the compiler
        :param solc_version: Compiler version required for compilation
        :return:
        """
        product = await compile_by_solcjs(
            standard_json=standard_json,
            solc_version=solc_version,
        )
        if product is None:
            return []
        idx2filename = {val['id']: fn for fn, val in product['sources'].items()}

        # extract yul function nodes
        extended_funcs = list()
        for contract in product['contracts'].values():
            for contract_name in contract.keys():
                yul = contract[contract_name].get('ir')
                if not yul or yul == '':
                    continue
                docs = contract[contract_name]['devdoc']['methods']
                docs.update(contract[contract_name]['userdoc']['methods'])
                yul_ast = await compile_yul_ast_by_solc(yul)
                objects = [
                    n for n in bfs_ast_node(yul_ast)
                    if n.get('nodeType') == 'YulObject' and
                       n.get('name').endswith('_deployed') and
                       n.get('name').startswith(contract_name)
                ]
                if len(objects) == 0:
                    continue
                obj = objects[0]['code']
                func_nodes = [
                    n for n in bfs_ast_node(obj)
                    if n.get('nodeType') == 'YulFunctionDefinition'
                ]

                name2doc = {key.split('(')[0]: value for key, value in docs.items()}
                name2func_node = {n.get('name'): n for n in func_nodes}

                # get all real src of given functions
                count = 1
                matched = set()
                for func in reversed(func_nodes):
                    if not func.get('name').startswith("fun_"):
                        continue
                    fun_name = re.sub(r'_\d*$', '', func.get('name')[4:])
                    if fun_name in matched:
                        continue
                    count += 1
                    func['sign'] = fun_name
                    doc = name2doc.get(fun_name)
                    if doc is None:
                        continue
                    matched.add(fun_name)

                    # match source code snippet
                    yul_begin_offset, yul_len, _ = func['nativeSrc'].split(':')
                    yul_begin_offset, yul_len = int(yul_begin_offset), int(yul_len)
                    source_filename = yul[yul_begin_offset - 300: yul_begin_offset] \
                        if sys.platform != 'win32' \
                        else yul.replace('\n', '\r\n')[yul_begin_offset - 300: yul_begin_offset]
                    source_filename = re.findall('@src (.*?):.*?:.*? ', source_filename)
                    if len(source_filename) == 0:
                        continue
                    source_filename = idx2filename[int(source_filename[-1])]
                    source_snippet = standard_json['sources'][source_filename]['content']
                    source_begin_offset, source_len, _ = func['src'].split(':')
                    source_begin_offset, source_len = int(source_begin_offset), int(source_len)
                    source_snippet = source_snippet[source_begin_offset: source_begin_offset + source_len]

                    # add extra data for func asts
                    func = copy.deepcopy(func)
                    func = extend_function(func, name2func_node)
                    func['source_code'] = source_snippet
                    func['documentation'] = str(doc)
                    func['whole_ir_code'] = yul
                    func['contract'] = contract_name
                    extended_funcs.append(func)
        return extended_funcs

    @staticmethod
    def save_ast2file(
            lock: multiprocessing.Semaphore,
            asts: List[Dict], path: str
    ):
        torch.save([YulASTDataset.ast2data(ast) for ast in asts], path)
        lock.release()

    @staticmethod
    def ast2data(ast: dict) -> HeteroData:
        q = [(0, ast)]
        ast_obj2index = {id(ast): 0}
        node_type2feats = {ast['nodeType']: [[
            0,
            *[int(n) for n in ast['nativeSrc'].split(':')],
        ]]}
        node_type2srcs = {ast['nodeType']: [[
            int(n) for n in ast['nativeSrc'].split(':')]
        ]}
        edge_type2edges = dict()
        edge_type2attrs = dict()
        opcode2idx = {opcode: i for i, opcode in enumerate(get_evm_opcodes())}
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
                q.append((depth + 1, sub_node))
                if not node_type2feats.get(sub_node['nodeType']):
                    node_type2feats[sub_node['nodeType']] = list()
                    node_type2srcs[sub_node['nodeType']] = list()

                idx = len(node_type2feats[sub_node['nodeType']])
                feats = [
                    depth + 1,
                    *[int(n) for n in sub_node['nativeSrc'].split(':')],
                ]
                if sub_node['nodeType'] == 'YulIdentifier':
                    vec = [0 for _ in range(len(opcode2idx) + 1)]
                    name = sub_node.get('name')
                    vec[opcode2idx.get(name, len(opcode2idx))] = 1
                    feats.extend(vec)

                node_type2feats[sub_node['nodeType']].append(feats)
                node_type2srcs[sub_node['nodeType']].append([
                    int(n) for n in sub_node['nativeSrc'].split(':')
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
        data['source_code'] = ast.get('source_code', '')
        data['ir_code'] = ast.get('whole_ir_code', '')
        data['documentation'] = ast.get('documentation', '')
        data['address'] = ast.get('address', '')
        data['contract'] = ast.get('contract', '')
        data['sign'] = ast.get('sign', '')
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
    from daos.transform import TransformSequence, format_prompt, format_data_type

    path = r'/home/my/lzw/CCIHunter/UniIntent/SmartCoCo2'
    curr = 1
    dataset = YulASTDataset(
        root=path, currency=curr,
        transform=TransformSequence([
            format_data_type,
            format_prompt,
            # format_ir,
            # functools.partial(
            #     embed_ir_feats,
            #     RoBERTa(tuning_layers=0, **model_args)
            # )
        ])
    )
    for i, data in enumerate(dataset):
        try:
            data.validate()
        except:
            print('error at: #%d' % i)
            traceback.print_exc()
