import os
import json
import re
import csv



def read_ast_files(filepath, docs):
    # 遍历目录中的文件
    try:
        # 读取文件内容
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        data = json.loads(content)
        for source in data['output']['sources'].values():
            q = [(None, source['ast'])]
            while len(q) > 0:
                contract, node = q.pop(0)
                if not isinstance(node, dict) or not node.get('nodeType'):
                    continue
                if node['nodeType'] == 'FunctionDefinition':
                    node['contract'] = contract
                    doc = node.get('documentation')
                    comment = doc if isinstance(doc, str) else doc.get('text', '') if isinstance(doc, dict) else ''
                    if not comment:
                        continue
                    key = contract + '-' + node.get('name') 
                    comment = re.sub(r'[\r\n]+', ' ', comment)
                    comment = comment.replace('@dev', '')
                    docs[key] = comment
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

    except Exception as e:
        print(f"Error processing file: {e}")


def update_doc(docs):
    pattern = re.compile(r'See \{(.*?)\}')
    updated_docs = dict(docs)
    for key, value in docs.items():
        matches = pattern.findall(value)
        for match in matches:
            reference_key = match
            if reference_key in docs:
                value = pattern.sub(docs[reference_key], value)
            else:
                # 合约自引用
                contract, _, _ = key.partition('-')
                reference_key2 = contract + '-' + match
                if reference_key2 in docs:
                    print(reference_key2)
                    value = pattern.sub(docs[reference_key2], value)            
        updated_docs[key] = value
    return updated_docs


file1 = '/home/my/lzw/CCIHunter/UniIntent/openzeppelin.json'
file2 = '/home/my/lzw/CCIHunter/UniIntent/openzeppelin_upgrade.json'
output_file = '/home/my/lzw/CCIHunter/UniIntent/standard_doc.csv'
docs = {}
read_ast_files(file1, docs)
read_ast_files(file2, docs)
print(len(docs))
updated_docs = update_doc(docs)
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for key, comment in updated_docs.items():
        writer.writerow([key, comment])
    print("OK")