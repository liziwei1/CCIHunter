import copy
import multiprocessing as mp
import random
from typing import List, Callable, Union, Tuple

import torch
from torch_geometric.data import HeteroData, Dataset, Batch

from utils.opcode import get_evm_opcodes

# Define lists of operators for different types
binary_operations = ['+', '-', '*', '/', '%', '**', '&&', '||', '!=', '==', '<', '<=', '>', '>=', '^', '&', '|', '<<', '>>']
unary_operations = ['-', '++', '--', '!', 'delete', '~']
assignment_operations = ['=', '+=', '-=', '*=', '%=', '/=', '|=' '&=', '^=', '>>=', '<<=']
statemutability = ["payable", "pure", "nonpayable", "view"]
visibility = ["external", "public", "internal", "private"]

binaryop2idx = {op: i for i, op in enumerate(binary_operations)}
unaryop2idx = {op: i for i, op in enumerate(unary_operations)}
assignop2idx = {op: i for i, op in enumerate(assignment_operations)}
stateop2idex = {op: i for i, op in enumerate(statemutability)}
visop2idx = {op: i for i, op in enumerate(visibility)}


class SolMutationDataset(Dataset):
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


class SolASTMutator:
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

'''
7 mutation operator
AOR: Assignment Operator Replacement
BOR Binary Operator Replacement
UOR: Unary Operator Replacement
PKR: Payable Keyword Replacement
FVR: Function Visibility Replacement
EED: event emssion deletion
MOD: Modifier Deletion
'''
def operator_replace(data: HeteroData, feat_name: str, operation_list: list, op_offset: int) -> Union[HeteroData, None]:
    if not data[feat_name]:
        return None
    feats = data[feat_name].x
    num_feats = len(feats)
    if num_feats < 1:
        return None

    # mutate randomly
    idx = random.choice(range(num_feats))
    feat = feats[idx]
    op_idx = int(torch.argmax(feat[op_offset:]))
    choices = [i for i in range(len(operation_list)) if i != op_idx]
    mutated_idx = random.choice(choices) if choices else None
    if mutated_idx is None:
        return None

    mutated_idx += op_offset
    feat[op_idx + op_offset] = 0
    feat[mutated_idx] = 1
    return data


def delete_node_and_edges(data: HeteroData, node_type: str) -> HeteroData:
    # Check if the node type exists in the data
    data_nodes = data[node_type]
    if not data_nodes:
        return None

    # Get the features of the nodes of the specified type
    node_feats = data_nodes.x
    num_feats = len(node_feats)
    # Randomly select a node
    idx = random.choice(range(num_feats))
    del_node = node_feats[idx]

    # Initialize a dictionary to keep track of nodes to be deleted
    del_dict = {nt: set() for nt in data.node_types}
    del_dict[node_type].add(idx)
    q = [(node_type, idx)]

    while len(q) > 0:
        node_type, node_index = q.pop(0)
        # Find edge types that originate from the node type
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
                if child in del_dict[et[2]]:
                    continue
                del_dict[et[2]].add(child)
                q.append((et[2], child))

    # Create a mask for each node type to indicate which nodes to delete
    del_masks = dict()
    for t, nodes in del_dict.items():
        del_mask = [True for _ in range(data[t].x.shape[0])]
        for idx in nodes:
            del_mask[idx] = False
        del_masks[t] = torch.tensor(del_mask)

    # Return the updated subgraph with nodes and edges deleted
    data = data.subgraph(del_masks)
    return data
    

def AOR(data: HeteroData) -> Union[HeteroData, None]:
    return operator_replace(data, 'Assignment', assignment_operations, 4)

def BOR(data: HeteroData) -> Union[HeteroData, None]:
    return operator_replace(data, 'BinaryOperation', binary_operations, 4)

def UOR(data: HeteroData) -> Union[HeteroData, None]:
    return operator_replace(data, 'UnaryOperation', unary_operations, 4)

def PKR(data: HeteroData) -> Union[HeteroData, None]:
    return operator_replace(data, 'FunctionDefinition', statemutability, 4)

def FVR(data: HeteroData) -> Union[HeteroData, None]:
    return operator_replace(data, 'FunctionDefinition', visibility, 8)

def EED(data: HeteroData) -> HeteroData:
    return delete_node_and_edges(data, 'EmitStatement')

def MOD(data: HeteroData) -> HeteroData:
    return delete_node_and_edges(data, 'ModifierInvocation')