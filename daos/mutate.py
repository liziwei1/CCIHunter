import copy
import multiprocessing as mp
import random
from typing import List, Callable, Union, Tuple

import torch
from torch_geometric.data import HeteroData, Dataset, Batch

from utils.opcode import get_evm_opcodes


class MutationDataset(Dataset):
    def __init__(self, mutator, dataset: Dataset):
        self.mutator = mutator
        self._dataset = dataset
        super().__init__(root=self._dataset.root)

    @property
    def metadata(self):
        return self._dataset.metadata

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._dataset.raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._dataset.processed_file_names

    def process(self):
        return

    def len(self) -> int:
        return len(self._dataset)

    def get(self, idx: int):
        data = self._dataset[idx]
        batch_data = self.mutator.mutate(data)
        batch_data = Batch.from_data_list(batch_data)
        return batch_data


class YulASTMutator:
    def __init__(
            self,
            mutate_funcs: List[Callable],
            max_mutant: int = 16,
    ):
        self.mutate_funcs = mutate_funcs
        self.max_mutant = max_mutant
        self._queue = mp.Queue()

    def mutate(self, data: HeteroData) -> List[HeteroData]:
        rlt = [data]
        for _ in range(self.max_mutant):
            mutate_func = random.choice(self.mutate_funcs)
            mutant = mutate_func(copy.deepcopy(data))
            if mutant is not None:
                rlt.append(mutant)
        return rlt


def mutate_operator(data: HeteroData) -> Union[HeteroData, None]:
    op_offset = 4
    feats = data['YulIdentifier'].x
    num_feats = len(feats)
    if num_feats < 1:
        return

    # init mutate ops
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

    # mutate randomly
    idx = random.choice(range(num_feats))
    feat = feats[idx]
    op_idx = int(torch.argmax(feat[op_offset:]))
    mutated_idx = mutation_ops.get(op_idx)
    if mutated_idx is None:
        return
    mutated_idx += op_offset
    data['YulIdentifier'].x[idx][op_idx] = 0
    data['YulIdentifier'].x[idx][mutated_idx] = 1
    return data


def mutate_stat_del(data: HeteroData) -> HeteroData:
    edge_type = [t for t in data.edge_types if t[1] == 'statements']
    edge_type = random.choice(edge_type)
    edge_store = data[edge_type]

    # select a stat randomly
    idx = random.randint(0, edge_store.edge_index.shape[1] - 1)
    node_index = edge_store.edge_index[0][idx].item()
    node_type = edge_type[0]

    # del stat and children nodes
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
    data = data.subgraph(del_masks)
    return data
