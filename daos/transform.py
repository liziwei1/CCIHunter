import copy
import random
import re
from typing import List, Callable, Dict

import torch.nn
from torch_geometric.data import HeteroData

from utils.naming import split_camel_case
from utils.opcode import get_evm_opcodes


class TransformSequence:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: HeteroData, *args, **kwargs):
        for transform in self.transforms:
            data = transform(data)
        return data


def format_data_type(data: HeteroData) -> HeteroData:
    for nt in data.node_types:
        data[nt].x = data[nt].x.float()
    for et in data.edge_types:
        data[et].edge_index = data[et].edge_index.long()
        data[et].edge_attr = data[et].edge_attr.float()
    return data


def format_prompt(data: HeteroData) -> HeteroData:
    prompt = "This is a function named `{}`, " \
             "and the corresponding document is the following: {}"
    sign = split_camel_case(re.sub('\(.*?\)', '', data['sign']))
    sign = ' '.join([word.lower() for word in sign])
    doc = data['documentation']
    data['prompt'] = prompt.format(sign, doc)
    return data

def sol_format_prompt(data: HeteroData) -> HeteroData:
    prompt = "This is a function named `{}`, " \
             "and the corresponding documentation is the following: \" {} \" "
    #sign = split_camel_case(re.sub('\(.*?\)', '', data['sign']))
    #sign = ' '.join([word.lower() for word in sign])
    doc = data.documentation
    final_prompt = prompt.format(data['sign'], doc)
    call_prompt1 = "It calls a function named `{}`.\n"
    call_prompt2 = "It calls a function named `{}`, which has the following comment: \" {} \" \n"
    if data.get('call'):
        final_prompt += "\nDuring its execution, \n"
        for call_func in data.get('call'):
            for name, comment in call_func.items():
                if not comment or comment == '':
                    final_prompt += call_prompt1.format(name)
                else:
                    final_prompt += call_prompt2.format(name, comment)
    data['prompt'] = final_prompt
    return data


def format_ir(data: HeteroData) -> HeteroData:
    ir = data['ir_code'].replace('\n', '\r\n')
    for nt in data.node_types:
        ir_codes = list()
        for src in data[nt].src:
            begin_offset, length = int(src[0]), int(src[1])
            code = ir[begin_offset: begin_offset + length]
            ir_codes.append(code)
        data[nt]['src_ir'] = ir_codes
    return data


def label_binding(label: Dict, data: HeteroData) -> HeteroData:
    filename, contract, sign = data['filename'].split('.')[0], data['contract'], data['sign']
    key = '{}@{}@{}'.format(filename, contract, sign)
    data['key'] = key
    data['label'] = label.get(key, True)
    return data

def fail_label_binding(label: Dict, data: HeteroData) -> HeteroData:
    filename, contract, sign = data['filename'].split('.')[0], data['contract'], data['sign']
    key = '{}@{}@{}'.format(filename, contract, sign)
    data['fail_label'] = label.get(key, False)
    return data


def del_mutate(data: HeteroData, node_type: str, node_index: int) -> HeteroData:
    del_dict = {nt: set() for nt in data.node_types}
    del_dict[node_type].add(node_index)
    q = [(node_type, node_index)]
    while len(q) > 0:
        node_type, node_index = q.pop(0)
        if node_index in del_dict[node_type]:
            continue
        edge_types = [
            t for t in data.edge_types
            if t[0] == node_type and t[1][0] != '_'
        ]
        for et in edge_types:
            mask = data[et].edge_index[0] == node_index
            children = data[et].edge_index[:, mask]
            children = children.t().tolist()
            children = [c[1] for c in children]
            for child in children:
                del_dict[et[2]].add(child)
                q.append((et[2], child))

    # return updated subgraph
    del_masks = dict()
    for t, nodes in del_dict.items():
        del_mask = [True for _ in range(data[t].x.shape[0])]
        for idx in nodes:
            del_mask[idx] = False
        del_masks[t] = torch.tensor(del_mask)
    return data.subgraph(del_masks)


def mutate(data: HeteroData) -> List[HeteroData]:
    # 定义变异上限
    # 每个data最多生成max_cnt1个由变换运算符生成的变异体
    # 每个data最多生成max_cnt2个由删除语句生成的变异体
    max_cnt1 = 8
    max_cnt2 = 7
    mutants = [data]

    # 定义逻辑运算符变异对
    op2idx = {op: i for i, op in enumerate(get_evm_opcodes())}
    mutation_ops = {
        op2idx['add']: op2idx['sub'],
        op2idx['mul']: op2idx['div'],
        op2idx['and']: op2idx['or'],
        op2idx['sgt']: op2idx['slt'],
        op2idx['gt']: op2idx['lt'],
        op2idx['sub']: op2idx['add'],
        op2idx['div']: op2idx['mul'],
        op2idx['or']: op2idx['and'],
        op2idx['slt']: op2idx['sgt'],
        op2idx['lt']: op2idx['gt']
    }

    # 变异1：运算符变异
    op_offset = 4
    feats = data['YulIdentifier'].x
    random_indices = list(range(len(feats)))
    random.shuffle(random_indices)
    while max_cnt1 > 0 and len(random_indices) > 0:
        max_cnt1 -= 1
        idx = random_indices.pop()
        feat = feats[idx]
        op_idx = int(torch.argmax(feat[op_offset:]))
        mutated_idx = mutation_ops.get(op_idx)
        if mutated_idx is None:
            continue
        mutated_idx += op_offset
        new_data = copy.deepcopy(data)
        new_data['YulIdentifier'].x[idx][op_idx] = 0
        new_data['YulIdentifier'].x[idx][mutated_idx] = 1
        mutants.append(new_data)

    # 变异2：随机语句删除
    # 随机选择max_cnt2个statement
    target = [t for t in data.edge_types if t[1] == 'statements']
    random.shuffle(target)
    while max_cnt2 > 0 and len(target) > 0:
        max_cnt2 -= 1
        e_type = target.pop()
        edges = data[e_type]
        length = len(edges['edge_index'][0])

        # 随机选择一条连边删除
        idx = random.randint(0, length - 1)
        node_type = e_type[0]
        node_index = edges['edge_index'][0][idx].item()
        mutants.append(del_mutate(data, node_type, node_index))
    return mutants
