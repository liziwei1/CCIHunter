from typing import Generator, List


def get_sub_nodes(node: dict) -> List:
    """
    Get sub ast nodes from a given ast node.

    :param node: an ast node
    :return: the sub node list
    """
    sub_nodes = list()
    for v in node.values():
        if isinstance(v, dict):
            sub_nodes.append(v)
        elif isinstance(v, list):
            sub_nodes.extend([item for item in v if not isinstance(item, int)])
    return sub_nodes


def bfs_ast_node(node: dict) -> Generator:
    """
    Scan all sub nodes using BFS.

    :param node: an ast node
    :return: a generator of sub node
    """
    seen_nodes = set()
    q = [node]
    while len(q) > 0:
        node = q.pop(0)
        if node is None or not isinstance(node, dict) or 'nodeType' not in node:
            continue

        if node.get('id') in seen_nodes:
            continue
        seen_nodes.add(node.get('id'))

        for sub_node in get_sub_nodes(node):
            q.append(sub_node)
        yield node
