import json
import os
import re
import sys
from typing import Dict, Union
from asyncio import subprocess

from settings import SOLC_PATH
from utils.tmpfile import wrap_run4tmpfile

_PLATFORM2SOLC_PATH = {
    'win32': os.path.join(SOLC_PATH, 'solc-windows-amd64-v0.8.21+commit.d9974bed.exe'),
    'linux': os.path.join(SOLC_PATH, 'solc-linux-amd64-v0.8.21+commit.d9974bed'),
    'darwin': os.path.join(SOLC_PATH, 'solc-macosx-amd64-v0.8.21+commit.d9974bed'),
}


async def compile_yul_ast_by_solc(
        content: str,
        **kwargs,
) -> Union[Dict, None]:
    """
    Compile the Tul AST using the solc file directly.

    :param content: Yul content.
    :return: a compacted result json.

    """
    return await wrap_run4tmpfile(
        data=content,
        async_func=lambda p: _solc_compile_with_file(p),
    )


async def _solc_compile_with_file(content_path: str) -> Union[Dict, None]:
    solc_path = _PLATFORM2SOLC_PATH.get(sys.platform)
    if solc_path is None:
        return None
    cmd = [solc_path, '--assemble', '--ast-compact-json']
    cmd.append(content_path)
    process = await subprocess.create_subprocess_shell(
        ' '.join(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, _ = await process.communicate()
    output = output.decode()
    rlt = re.search(r'\{.*\}', output)
    if rlt is None:
        return None
    product = json.loads(rlt.group())
    return product
