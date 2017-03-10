#!/usr/bin/env python2

import unittest
from ProvenanceGraphBuilding import *

class TestGraphFuncs(unittest.TestCase):
    def setUp(self):
        self.basic_test_edges_1 = [(0, 1), (1, 2), (1, 4), (3, 4), (4, 5)]
        self.basic_test_edges_2 = [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)]
        self.basic_test_edges_3 = [(1, 0), (1, 3), (2, 3), (3, 4)]

class TestMiscFuncs(TestGraphFuncs):
    def test_group_by_fun(self):
        grouped_edges_1 = { 0: [(0, 1)], 1: [(1, 2), (1, 4)], 3: [(3, 4)], 4: [(4, 5)] }
        grouped_edges_2 = { 1: [(0, 1)], 2: [(1, 2)], 4: [(1, 4), (3, 4)], 5: [(4, 5)] }
        self.assertEqual(group_by_fun(lambda e: e[0], self.basic_test_edges_1), grouped_edges_1)
        self.assertEqual(group_by_fun(lambda e: e[1], self.basic_test_edges_1), grouped_edges_2)

class TestGraphReducerFuncs(TestGraphFuncs):
    def setUp(self):
        super(TestGraphReducerFuncs, self).setUp()
        self.basic_test_edges_wpath_1 = [ EdgeRecord(t[0], t[1], Path(t, None)) for t in self.basic_test_edges_1 ]
        self.basic_test_edges_wpath_2 = [ EdgeRecord(t[0], t[1], Path(t, None)) for t in self.basic_test_edges_2 ]
        self.basic_test_edges_wpath_3 = [ EdgeRecord(t[0], t[1], Path(t, None)) for t in self.basic_test_edges_3 ]
    
    def test_reduce_graph(self):
        selected_nodes_1 = [0, 2, 3, 5]
        reduced_graph_1 = {EdgeRecord(0, 5, Path(edge=(0, 1), rest=Path(edge=(1, 4), rest=Path(edge=(4, 5), rest=None)))),
                           EdgeRecord(0, 2, Path(edge=(0, 1), rest=Path(edge=(1, 2), rest=None))),
                           EdgeRecord(3, 5, Path(edge=(3, 4), rest=Path(edge=(4, 5), rest=None)))}

        selected_nodes_2 = [0, 1, 2, 3, 6]
        reduced_graph_2 = {EdgeRecord(0, 6, Path(edge=(0, 4), rest=Path(edge=(4, 6), rest=None))),
                           EdgeRecord(1, 6, Path(edge=(1, 4), rest=Path(edge=(4, 6), rest=None))),
                           EdgeRecord(2, 6, Path(edge=(2, 5), rest=Path(edge=(5, 6), rest=None))),
                           EdgeRecord(3, 6, Path(edge=(3, 5), rest=Path(edge=(5, 6), rest=None)))}

        selected_nodes_2_1 = [1, 2, 3, 6]
        reduced_graph_2_1 = {EdgeRecord(1, 6, Path(edge=(1, 4), rest=Path(edge=(4, 6), rest=None))),
                             EdgeRecord(2, 6, Path(edge=(2, 5), rest=Path(edge=(5, 6), rest=None))),
                             EdgeRecord(3, 6, Path(edge=(3, 5), rest=Path(edge=(5, 6), rest=None)))}

        self.assertEqual(reduce_graph(self.basic_test_edges_wpath_1, selected_nodes_1), reduced_graph_1)
        self.assertEqual(reduce_graph(self.basic_test_edges_wpath_2, selected_nodes_2), reduced_graph_2)
        self.assertEqual(reduce_graph(self.basic_test_edges_wpath_2, selected_nodes_2_1), reduced_graph_2_1)

    def test_build_direct_graph(self):
        probe_node_1_1 = 2
        probe_node_1_2 = 3
        probe_node_1_3 = 5
        probe_node_1_4 = 6
        
        direct_graph_1_1 = {EdgeRecord(2, 5, Path(edge=(2, 5), rest=None)),
                            EdgeRecord(5, 6, Path(edge=(5, 6), rest=None))}
        
        direct_graph_1_2 = {EdgeRecord(3, 5, Path(edge=(3, 5), rest=None)),
                            EdgeRecord(5, 6, Path(edge=(5, 6), rest=None))}
        
        direct_graph_1_3 = {EdgeRecord(2, 5, Path(edge=(2, 5), rest=None)),
                            EdgeRecord(3, 5, Path(edge=(3, 5), rest=None)),
                            EdgeRecord(5, 6, Path(edge=(5, 6), rest=None))}
        
        direct_graph_1_4 = {EdgeRecord(0, 4, Path(edge=(0, 4), rest=None)),
                            EdgeRecord(1, 4, Path(edge=(1, 4), rest=None)),
                            EdgeRecord(2, 5, Path(edge=(2, 5), rest=None)),
                            EdgeRecord(3, 5, Path(edge=(3, 5), rest=None)),
                            EdgeRecord(4, 6, Path(edge=(4, 6), rest=None)),
                            EdgeRecord(5, 6, Path(edge=(5, 6), rest=None))}

        probe_node_2_1 = 1
        probe_node_2_2 = 3

        direct_graph_2_1 = {EdgeRecord(1, 0, Path(edge=(1, 0), rest=None)),
                            EdgeRecord(1, 3, Path(edge=(1, 3), rest=None)),
                            EdgeRecord(3, 4, Path(edge=(3, 4), rest=None))}

        direct_graph_2_2 = {EdgeRecord(1, 3, Path(edge=(1, 3), rest=None)),
                            EdgeRecord(2, 3, Path(edge=(2, 3), rest=None)),
                            EdgeRecord(3, 4, Path(edge=(3, 4), rest=None))}
        
        probe_node_2 = 99

        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_2, probe_node_1_1), direct_graph_1_1)
        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_2, probe_node_1_2), direct_graph_1_2)
        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_2, probe_node_1_3), direct_graph_1_3)
        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_2, probe_node_1_4), direct_graph_1_4)

        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_3, probe_node_2_1), direct_graph_2_1)
        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_3, probe_node_2_2), direct_graph_2_2)

        self.assertEqual(build_direct_graph(self.basic_test_edges_wpath_2, probe_node_2), set())

class TestGraphPathFuncs(TestGraphFuncs):
    def setUp(self):
        super(TestGraphPathFuncs, self).setUp()
        self.p1 = Path((0, 1), None)
        self.p2 = Path((1, 2), None)        
        self.p1_p2 = Path((0, 1), Path((1, 2), None))
        self.p2_p1 = Path((1, 2), Path((0, 1), None))

        self.p3 = Path((2, 3), None)
        self.p1_p2_p3 = Path((0, 1), Path((1, 2), Path((2, 3), None)))
    
    def test_append_to_path(self):
        self.assertEqual(append_to_path(self.p1, self.p2), self.p1_p2)
        self.assertEqual(append_to_path(self.p1_p2, self.p3), self.p1_p2_p3)

class TestEdgeFiltering(unittest.TestCase):
    def setUp(self):
        self.filter_a = lambda e: e == "A"
        self.filter_b = lambda e: e == "B"

        self.el_filter_c = lambda el: map(lambda e: e == "C", el)

        self.edge_list_1 = ["A", "B", "C", "C", "A"]

    def test_reject_edges(self):
        self.assertEqual(reject_edges(self.edge_list_1), [False, False, False, False, False])
        self.assertEqual(reject_edges([]), [])
        self.assertEqual(reject_edges(self.edge_list_1, [self.el_filter_c]), [False, False, True, True, False])
        self.assertEqual(reject_edges(self.edge_list_1, [self.el_filter_c], [self.filter_a, self.filter_b]), [True, True, True, True, True])
        self.assertEqual(reject_edges(self.edge_list_1, [], [self.filter_a, self.filter_b]), [True, True, False, False, True])

class TestDetectCycle(unittest.TestCase):
    def setUp(self):
        self.graph_0 = set()
        self.graph_1 = {EdgeRecord(0, 4, Path(edge=(0, 4), rest=None)),
                        EdgeRecord(1, 4, Path(edge=(1, 4), rest=None)),
                        EdgeRecord(2, 5, Path(edge=(2, 5), rest=None)),
                        EdgeRecord(3, 5, Path(edge=(3, 5), rest=None)),
                        EdgeRecord(4, 6, Path(edge=(4, 6), rest=None)),
                        EdgeRecord(5, 6, Path(edge=(5, 6), rest=None))}
        self.graph_2 = {EdgeRecord(0, 1, Path(edge=(0, 1), rest=None)),
                        EdgeRecord(1, 0, Path(edge=(1, 0), rest=None))}
        self.graph_3 = {EdgeRecord(0, 1, Path(edge=(0, 1), rest=None)),
                        EdgeRecord(1, 2, Path(edge=(1, 2), rest=None)),
                        EdgeRecord(2, 3, Path(edge=(2, 3), rest=None)),
                        EdgeRecord(3, 1, Path(edge=(3, 1), rest=None))}
        self.graph_4 = {EdgeRecord(0, 4, Path(edge=(0, 4), rest=None)),
                        EdgeRecord(1, 4, Path(edge=(1, 4), rest=None)),
                        EdgeRecord(2, 5, Path(edge=(2, 5), rest=None)),
                        EdgeRecord(3, 5, Path(edge=(3, 5), rest=None)),
                        EdgeRecord(4, 6, Path(edge=(4, 6), rest=None)),
                        EdgeRecord(5, 6, Path(edge=(5, 6), rest=None)),
                        EdgeRecord(5, 7, Path(edge=(5, 7), rest=None)),
                        EdgeRecord(7, 8, Path(edge=(7, 8), rest=None)),
                        EdgeRecord(8, 5, Path(edge=(8, 5), rest=None))}

    def test_detect_cycle(self):
        self.assertFalse(detect_cycle(self.graph_0))
        self.assertFalse(detect_cycle(self.graph_1))
        self.assertTrue(detect_cycle(self.graph_2))
        self.assertTrue(detect_cycle(self.graph_3))
        self.assertTrue(detect_cycle(self.graph_4))
            
if __name__ == '__main__':
    unittest.main()
