import numpy as np

def set_similarity_overlap(set_1, set_2):
    return 2 * np.float64(len(set_1 & set_2)) / (len(set_1) + len(set_2))

def SimNLO(nodeset_1, edgeset_1, nodeset_2, edgeset_2):
    return 2 * np.float64(len(nodeset_1 & nodeset_2) + len(edgeset_1 & edgeset_2)) / (len(nodeset_1) + len(nodeset_2) + len(edgeset_1) + len(edgeset_2))

def SimNO(nodeset_1, nodeset_2):
    return set_similarity_overlap(nodeset_1, nodeset_2)

def SimLO(edgeset_1, edgeset_2):
    return set_similarity_overlap(edgeset_1, edgeset_2)

def node_recall(ref_nodeset, sys_nodeset):
    return np.float64(len(ref_nodeset & sys_nodeset)) / len(ref_nodeset)
