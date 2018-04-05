#!/usr/bin/env python2

import unittest
import numpy as np

from ProvenanceMetrics import *

def _set_to_dict(s):
    return { k: k for k in s }

def _minimapping(a, b):
    return [ (k, a.get(k, None), b.get(k, None)) for k in set(a) | set(b) ]

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.nodeset_1 = _set_to_dict(set())
        self.nodeset_2 = _set_to_dict({ 0, 1, 2, 3 })
        self.nodeset_3 = _set_to_dict({ 0, 1, 2 })
        self.nodeset_4 = _set_to_dict({ 3 })

        self.edgeset_1 = _set_to_dict(set())
        self.edgeset_2 = _set_to_dict({ (0,1), (1,2), (2,3) })
        self.edgeset_3 = _set_to_dict({ (0,1), (1,2) })
        self.edgeset_4 = _set_to_dict({ (2, 3) })

    def test_SimNLO(self):
        self.assertTrue(np.isnan(SimNLO(_minimapping(self.nodeset_1, self.nodeset_1), _minimapping(self.edgeset_1, self.edgeset_1))))

        self.assertEqual(SimNLO(_minimapping(self.nodeset_1, self.nodeset_2), _minimapping(self.edgeset_1, self.edgeset_2)), 0)
        self.assertEqual(SimNLO(_minimapping(self.nodeset_2, self.nodeset_2), _minimapping(self.edgeset_2, self.edgeset_2)), 1)

        self.assertEqual(SimNLO(_minimapping(self.nodeset_2, self.nodeset_3), _minimapping(self.edgeset_2, self.edgeset_3)), 2 * 5 / float(12))
        self.assertEqual(SimNLO(_minimapping(self.nodeset_3, self.nodeset_4), _minimapping(self.edgeset_3, self.edgeset_4)), 0)
        self.assertEqual(SimNLO(_minimapping(self.nodeset_2, self.nodeset_4), _minimapping(self.edgeset_2, self.edgeset_4)), 2 * 2 / float(9))

    def test_set_similarity_overlap(self):
        self.assertTrue(np.isnan(set_similarity_overlap(_minimapping(self.nodeset_1, self.nodeset_1))))

        self.assertEqual(set_similarity_overlap(_minimapping(self.nodeset_1, self.nodeset_2)), 0)
        self.assertEqual(set_similarity_overlap(_minimapping(self.nodeset_2, self.nodeset_2)), 1)

        self.assertEqual(set_similarity_overlap(_minimapping(self.nodeset_2, self.nodeset_3)), 2 * 3 / float(7))
        self.assertEqual(set_similarity_overlap(_minimapping(self.edgeset_2, self.edgeset_3)), 2 * 2 / float(5))
        self.assertEqual(set_similarity_overlap(_minimapping(self.nodeset_3, self.nodeset_4)), 0)
        self.assertEqual(set_similarity_overlap(_minimapping(self.edgeset_3, self.edgeset_4)), 0)
        self.assertEqual(set_similarity_overlap(_minimapping(self.nodeset_2, self.nodeset_4)), 2 * 1 / float(5))
        self.assertEqual(set_similarity_overlap(_minimapping(self.edgeset_2, self.edgeset_4)), 2 * 1 / float(4))

    def test_node_recall(self):
        self.assertTrue(np.isnan(node_recall(_minimapping(self.nodeset_1, self.nodeset_2))))

        self.assertEqual(node_recall(_minimapping(self.nodeset_2, self.nodeset_3)), 3 / float(4))
        self.assertEqual(node_recall(_minimapping(self.nodeset_3, self.nodeset_2)), 3 / float(3))
        self.assertEqual(node_recall(_minimapping(self.nodeset_2, self.nodeset_2)), 1)
        self.assertEqual(node_recall(_minimapping(self.nodeset_3, self.nodeset_4)), 0)

if __name__ == '__main__':
    unittest.main()
