import numpy as np

def _ref_selector(t):
    k, r, s = t
    return (r != None)

def _sys_selector(t):
    k, r, s = t
    return (s != None)

def _corr_selector(t):
    k, r, s = t
    return (r != None and s != None)

def _fa_selector(t):
    k, r, s = t
    return (r == None and s != None)

def _miss_selector(t):
    k, r, s = t
    return (r != None and s == None)

def num_ref(mapping):
    return len(filter(_ref_selector, mapping))

def num_sys(mapping):
    return len(filter(_sys_selector, mapping))

def num_corr(mapping):
    return len(filter(_corr_selector, mapping))

def num_miss(mapping):
    return len(filter(_miss_selector, mapping))

def num_fa(mapping):
    return len(filter(_fa_selector, mapping))

def set_similarity_overlap(mapping):
    return 2 * np.float64(num_corr(mapping)) / (num_ref(mapping) + num_sys(mapping))

def SimNLO(node_mapping, edge_mapping):
    return 2 * np.float64(num_corr(node_mapping) + num_corr(edge_mapping)) / (num_ref(node_mapping) + num_sys(node_mapping) + num_ref(edge_mapping) + num_sys(edge_mapping))

def SimNO(node_mapping):
    return set_similarity_overlap(node_mapping)

def SimLO(edge_mapping):
    return set_similarity_overlap(edge_mapping)

def node_recall(node_mapping):
    return np.float64(num_corr(node_mapping)) / num_ref(node_mapping)
