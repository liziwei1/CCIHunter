const solc_ast = require('solc-typed-ast');
const ASTReader = solc_ast['ASTReader'];

const solc_ast_json_data = %s; // Important! Don't modify it!
var config;
if (solc_ast_json_data['children']) {
    const solc_ast_conf = require('solc-typed-ast/dist/ast/legacy/configuration');
    config = solc_ast_conf['LegacyConfiguration'];
} else {
    const solc_ast_conf = require('solc-typed-ast/dist/ast/modern/configuration');
    config = solc_ast_conf['ModernConfiguration'];
}

const reader = new ASTReader();
const root = reader.convert(solc_ast_json_data, config);
const allowed_node_types = {
    'FunctionDefinition': true,
}

let q = [];
let vis = new Set();
q.push(root);
while (q.length > 0) {
    let node = q.pop();
    if (vis.has(node.id)){
        continue;
    }
    vis.add(node.id);

    let children = node.getChildren();
    for (let i = 0; i < children.length; i++) {
        let child = children[i];
        if (!allowed_node_types[child.type]) {
            continue;
        }
        q.push(child);
    }

    if (!allowed_node_types[node.type]){
        continue;
    }
    var params = [];
    if(node['vParameters'] !== undefined) {
        for (const param of node['vParameters']['ownChildren']) {
            params.push(param['typeString']);
        }
    }
    let item = {
        'id': node.id,
        'src': node.src,
        'type': node.type,
        'sign': node['name'] + '(' + params.join(',') + ')',
    };
    console.log(JSON.stringify(item));
}
