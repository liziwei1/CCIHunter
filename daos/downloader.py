import copy
import json
import os
import re
import urllib.parse
from typing import Dict, List

from compiler import compile_by_solcjs
from compiler.solc import compile_yul_ast_by_solc
from ir.yul.doc_mapping import locate_source_src_from_yul
from ir.yul.function import extend_function
from ir.yul.scan import bfs_ast_node
from settings import CACHE_DIR
from typing import Dict
import aiohttp

from utils.standard_json import get_sol_standard_json
from utils.typed_ast import get_typed_ast_items


class Downloader:
    async def download(self, *args, **kwargs):
        result = await self._preprocess(*args, **kwargs)
        if result is not None:
            return result
        result = await self._fetch(*args, **kwargs)
        return await self._process(result, **kwargs)

    async def _preprocess(self, *args, **kwargs):
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        raise NotImplemented()

    async def _process(self, result, *args, **kwargs):
        raise NotImplemented()


class JSONRPCDownloader(Downloader):
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url

    def get_request_param(self, *args, **kwargs) -> Dict:
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        params = self.get_request_param(*args, **kwargs)
        print(params)
        client = aiohttp.ClientSession()
        async with client.post(**params) as response:
            rlt = await response.text()
        await client.close()
        return rlt


class EtherscanDownloader(Downloader):
    def __init__(self, apikey: str):
        self.apikey = apikey

    def get_request_param(self, *args, **kwargs) -> Dict:
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        params = self.get_request_param(*args, **kwargs)
        client = aiohttp.ClientSession()
        async with client.get(**params) as response:
            rlt = await response.text()
        await client.close()
        return rlt


class ContractSourceDownloader(EtherscanDownloader):
    def get_request_param(self, contract_address: str) -> Dict:
        query_params = urllib.parse.urlencode({
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address.lower(),
        })
        return {"url": '{}&{}'.format(self.apikey, query_params)}

    async def _preprocess(self, contract_address: str, **kwargs):
        path = os.path.join(CACHE_DIR, 'source', '%s.json' % contract_address)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = result['result'][0]

        # cache data
        contract_address = kwargs['contract_address']
        path = os.path.join(CACHE_DIR, 'source')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % contract_address)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result


async def test(address: str):
    d = ContractSourceDownloader('https://api.etherscan.io/api?apikey=7MM6JYY49WZBXSYFDPYQ3V7V3EMZWE4KJK')
    result = await d.download(
        contract_address=address,
    )
    save_name = address + ".json"
    save_path = r"D:\Dapp\UniIntent\label_data2\\" + save_name
    print(save_path)
    with open(save_path, 'w') as f:
        json.dump(result, f)
        print("save OK")


if __name__ == '__main__':
    import asyncio
    import pandas as pd

    path = r'D:\Dapp\UniIntent\manually labeled.csv'
    # 使用pandas读入
    data = pd.read_csv(path)  # 读取文件中所有数据
    all_address = []
    for index, (address, label) in data.iloc[:, [0, 6]].iterrows():
        address = str(address).split('.')[0]
        if address not in all_address:
            all_address.append(address)
            asyncio.run(test(address))
    #print(len(all_address))
