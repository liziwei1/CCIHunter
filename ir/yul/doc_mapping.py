import re
import sys
from typing import Tuple, Union


def locate_source_src_from_yul(
        yul_code: str,
        yul_begin_offset: int
) -> Union[Tuple[int, int, int], None]:
    """
    Get source src from yul code.
    Specially, we use the `nativeSrc` begin offset to locate
    the yul function string start index.
    And reserving scan the `@src` line indicate the source
    code range.
    Note that the `nativeSrc` format as `begin_offset`:`length`:0,
    and the `begin_offset` = `chars` + `lines`.

    :param yul_code:
    :param yul_begin_offset:
    :return: file_index, sol_begin_offset, sol_len
    """

    # PLEASE DO NOT DELETE THE IF STATEMENTS!
    # FK THE SOLC COMPILER!
    if sys.platform == 'win32':
        yul_code = yul_code.replace('\n', '\r\n')

    break_lines, rev_len = 0, 1
    while break_lines < 2 or yul_begin_offset - rev_len < 0:
        if yul_code[yul_begin_offset - rev_len] == '\n':
            break_lines += 1
        rev_len += 1
    s = yul_code[yul_begin_offset - rev_len: yul_begin_offset]
    rlt = re.search('@src (.*?):(.*?):(.*?) ', s)
    if rlt is None:
        return
    file_index = int(rlt.groups()[0])
    sol_begin_offset = int(rlt.groups()[1])
    sol_len = int(rlt.groups()[2]) - sol_begin_offset
    return sol_begin_offset, sol_len, file_index
