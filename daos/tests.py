import unittest

from daos.yul_ast import YulASTDataset


class DatasetTest(unittest.TestCase):
    def test_yul_ast_dataset(self):
        path = r'C:\Users\HP\Downloads\dappcons_dataset'
        dataset = YulASTDataset(root=path)
        self.assertTrue(len(dataset) > 0)
