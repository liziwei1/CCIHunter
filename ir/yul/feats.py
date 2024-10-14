from typing import Dict, List, Callable


class YulASTNodeFeatureBuilder:
    def __init__(self):
        self._build_map = {
            "YulLeave",
            "YulBreak",
            "YulFunctionCall",
            "YulLiteral",
            "YulIf",
            "YulVariableDeclaration",
            "YulExpressionStatement",
            "YulCase",
            "YulFunctionDefinition",
            "YulAssignment",
            "YulSwitch",
            "YulIdentifier",
            "YulForLoop",
            "YulContinue",
            "YulTypedName",
            "YulBlock"
        }

    def build_feats(self, node: Dict) -> List:
        node_type = node.get('nodeType')
        if node_type is None:
            return []

        func = self._build_map[node_type]
        return func(node)

    def _build(self, funcs: List[Callable], node: Dict) -> List:
        rlt = list()
        for func in funcs:
            rlt.extend(func(node))
        return rlt

    def build_basic_feats(self, node: Dict) -> List:
        pass


# Define lists of operators for different types
binary_operations = ['+', '-', '*', '/', '%', '**', '&&', '||', '!=', '==', '<', '<=', '>', '>=', '^', '&', '|', '<<', '>>']
unary_operations = ['-', '++', '--', '!', 'delete', '~']
assignment_operations = ['=', '+=', '-=', '*=', '%=', '/=', '|='  '&=', '^=', '>>=', '<<=']
statemutability = ["payable", "pure", "nonpayable", "view"]
visibility = ["external", "public", "internal", "private"]

def get_sol_feats(node: Dict) -> List[int]:
    """
    Generate feature encoding based on the type and operator of the node.
    
    :param node: Dictionary containing the operator and node type
    :return: Corresponding feature encoding list
    """
    nodetype = node.get('nodeType')
    operator = node.get('operator')
    
    if nodetype == 'BinaryOperation':
        return operator_to_onehot(operator, binary_operations)
    elif nodetype == 'UnaryOperation':
        return operator_to_onehot(operator, unary_operations)
    elif nodetype == 'Assignment':
        return operator_to_onehot(operator, assignment_operations)
    elif nodetype == 'FunctionDefinition':
        # [statemutability * 4, visibility * 4]
        state = node.get('stateMutability')
        vis = node.get('visibility')
        state_encoding = operator_to_onehot(state, statemutability)
        vis_encoding = operator_to_onehot(vis, visibility)
        state_encoding.extend(vis_encoding)
        return state_encoding
    else:
        return None
    
    


def operator_to_onehot(operator, operation_list):
    """
    Convert an operator to a one-hot encoding.
    
    :param operator: The operator string
    :param operation_list: List of all possible operators
    :return: Corresponding one-hot encoding list
    """
    # Initialize one-hot encoding list
    onehot_encoding = [0] * len(operation_list)
    # Find the index of the operator in the operation_list
    if operator in operation_list:
        index = operation_list.index(operator)
        # Set the corresponding index position to 1
        onehot_encoding[index] = 1
    return onehot_encoding