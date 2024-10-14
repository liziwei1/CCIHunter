import copy
import csv
import re
from typing import Dict

from ir.yul.scan import bfs_ast_node


def extend_function(
        func_node: dict, name2func_node: dict,
        cur_depth: int = 0, max_depth: int = 2,
) -> Dict:
    """
    Get the extended ast function node.
    Specially, the function call node in a function node will be replaced
    as the called function node (i.e., extend).

    :param func_node: the given yul function node
    :param name2func_node: a mapping from function name to function nodes
    :return: an extended function
    """
    for n in bfs_ast_node(func_node):
        if n.get('nodeType') != 'YulFunctionCall':
            continue
        name = n['functionName'].get('name')
        call_function = name2func_node.get(name)
        if call_function is None:
            continue

        # extend call function
        call_function = copy.deepcopy(call_function)
        if cur_depth < max_depth:
            call_function = extend_function(
                call_function, name2func_node,
                cur_depth + 1, max_depth,
            )
        
        n.clear()
        for k, v in call_function.items():
            n[k] = v
    return func_node


def sol_extend_function(
        func_node: dict, id2func_node: dict,
        cur_depth: int = 0, max_depth: int = 2,
) -> Dict:
    """
    Get the extended ast function node.
    Specially, the function call node in a function node will be replaced
    as the called function node (i.e., extend).

    :param func_node: the given sol function node
    :param name2func_node: a mapping from function name to function nodes
    :return: an extended function
    """
    for n in bfs_ast_node(func_node):
        if n.get('nodeType') != 'FunctionCall':
            continue
        # get the id of called function
        id = n['expression'].get('referencedDeclaration')
        call_function = id2func_node.get(id)
        if id is None or call_function is None:
            continue

        # extend call function
        call_function = copy.deepcopy(call_function)
        if cur_depth < max_depth:
            call_function = extend_function(
                call_function, id2func_node,
                cur_depth + 1, max_depth,
            )
        # n.clear()

        call_doc = {call_function.get('name') : call_function.get('doc')}
        if not any(call_doc == existing_call for existing_call in func_node['call']):
            func_node['call'].append(call_doc)
        for k, v in call_function.items():
            n[k] = v
    return func_node


def doc_ref(comment : str):
    # check ref
    pattern = re.compile(r'See \{(.*?)\}')
    matches = pattern.findall(comment)
    csv_file = 'standard_doc.csv'
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for match in matches:
            for row in reader:
                if row[0] == match:
                    # update comment
                    comment = pattern.sub(row[1][:-1], comment)
                    continue
    return comment
            
        
        
