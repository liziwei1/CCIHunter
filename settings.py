import os

PROJECT_PATH, _ = os.path.split(os.path.realpath(__file__))
MISC_PATH = os.path.join(PROJECT_PATH, 'misc')

SOLCJS_CODE_RELATED_PATH = os.path.join(MISC_PATH, 'solcjs.js')
with open(os.path.join(PROJECT_PATH, SOLCJS_CODE_RELATED_PATH), 'r') as f:
    SOLCJS_CODE = f.read()
TYPED_AST_CODE_RELATED_PATH = os.path.join(MISC_PATH, 'typed_ast.js')
with open(os.path.join(PROJECT_PATH, TYPED_AST_CODE_RELATED_PATH), 'r') as f:
    TYPED_AST_CODE = f.read()

SOLC_PATH = os.path.join(MISC_PATH, 'solc')

TMP_FILE_DIR = os.path.join(PROJECT_PATH, 'tmp')
if not os.path.exists(TMP_FILE_DIR):
    os.makedirs(TMP_FILE_DIR)

NODE_PATH = 'node'

HUGGING_MODEL_PATH = os.path.join(PROJECT_PATH, 'hugging_cache')
if not os.path.exists(HUGGING_MODEL_PATH):
    HUGGING_MODEL_PATH = None

CACHE_DIR = os.path.join(PROJECT_PATH, 'data/cache')