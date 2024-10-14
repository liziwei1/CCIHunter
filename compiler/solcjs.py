import json
from asyncio import subprocess
from typing import Dict, Union

from settings import SOLCJS_CODE, NODE_PATH
from utils.tmpfile import wrap_run4tmpfile


async def compile_by_solcjs(
        standard_json: Dict,
        solc_version: str,
) -> Union[Dict, None]:
    """
    Compile the solidity smart contract by standard json,
    please ensure the standard json set with outputting
    `userdoc` and `devdoc`.

    :param standard_json: the complied the standard json
    :param solc_version: solc compiler version
    :return: a `CompileResult`
    """
    if solc_version.startswith("0.4"):
        solc_version = "0.5.0"

    product = await wrap_run4tmpfile(
        data=SOLCJS_CODE % (solc_version, json.dumps(standard_json)),
        async_func=lambda p: _solcjs_compile_with_file(p),
    )


    if product is None or \
            product.get('contracts') is None or product.get('sources') is None:

        return None


    return product


async def _solcjs_compile_with_file(standard_json_path: Dict) -> Union[Dict, None]:
    cmd = [NODE_PATH, standard_json_path]
    try:
        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, _ = await process.communicate()
        product = json.loads(output.decode())
        return product
    except:
        return None
