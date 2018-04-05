import collections

def group_by_fun(fun, inlist):
    d = {}
    [ d.setdefault(fun(x), []).append(x) for x in inlist ]
    return d

Path = collections.namedtuple('Path', [ 'edge', 'rest' ])
def append_to_path(path, to_append):
    if path.rest is None:
        return Path(path.edge, to_append)
    else:
        return Path(path.edge, append_to_path(path.rest, to_append))

def path_to_list(path):
    def append_path_to_list(in_list, p):
        if p is None:
            return in_list
        else:
            in_list.append(p.edge)
            return append_path_to_list(in_list, p.rest)
    return append_path_to_list([], path)

EdgeRecord = collections.namedtuple('EdgeRecord', [ 'source', 'target', 'path' ])

# Where edge_records is a set of EdgeRecords and nodes and
# selected_nodes is a lists of node_id(s) to keep
def reduce_graph(edge_records, selected_nodes):
    edge_set = set(edge_records)
    ins = group_by_fun(lambda e: e.target, edge_set)
    outs = group_by_fun(lambda e: e.source, edge_set)

    def join_edges(in_edges, out_edges):
        new_edges = []
        for i in in_edges:
            for o in out_edges:
                new_edges.append(EdgeRecord(i.source, o.target, append_to_path(i.path, o.path)))
        return new_edges

    to_rm = (set(ins.keys()) | set(outs.keys())) - set(selected_nodes)
    if len(to_rm) == 0:
        return edge_set
    else:
        node_to_rm = to_rm.pop()
        new_edges = join_edges(ins.get(node_to_rm, []), outs.get(node_to_rm, []))
        return reduce_graph((edge_set - (set(ins.get(node_to_rm, [])) | set(outs.get(node_to_rm, [])))) | set(new_edges), selected_nodes)

def build_direct_graph(edge_records, probe_node):
    edge_set = set(edge_records)
    ins = group_by_fun(lambda e: e.target, edge_set)
    outs = group_by_fun(lambda e: e.source, edge_set)

    out_edges = set()
    up_nodes = {probe_node}
    while len(up_nodes) > 0:
        up_edges = reduce(lambda x, y: x | y, map(lambda n: set(ins.get(n, [])), up_nodes), set())
        out_edges |= up_edges
        up_nodes = set(map(lambda e: e.source, up_edges))

    down_nodes = {probe_node}
    while len(down_nodes) > 0:
        down_edges = reduce(lambda x, y: x | y, map(lambda n: set(outs.get(n, [])), down_nodes), set())
        out_edges |= down_edges
        down_nodes = set(map(lambda e: e.target, down_edges))

    return out_edges

# "edge_list_filters" argument should be an enumerable of functions
# which take a list of edges and return an equal length enumerable of
# boolean (True if the edge at a particular index should be filtered out)
# "edge_filters" argument should be an enumerable of functions which
# take an edge and return a boolean (True if that particular edge
# should be filtered out)
def reject_edges(edge_list, edge_list_filters=[], edge_filters=[]):
    def composed_edge_list_filter(edge_list):
        return map(lambda edge: any(map(lambda f: f(edge), edge_filters)), edge_list)

    edge_list_filters.append(composed_edge_list_filter)
    return map(any, zip(*map(lambda f: f(edge_list), edge_list_filters)))

def detect_cycle(edge_records):
    edge_set = set(edge_records)
    ins = group_by_fun(lambda e: e.target, edge_set)
    outs = group_by_fun(lambda e: e.source, edge_set)
    roots = { k for k in outs } - { k for k in ins }

    # Handle "rootless" case
    if len(roots) == 0 and len(edge_set) > 0:
        return True

    stack = []
    current_path = []
    current_visited = set()
    block_list = set()

    for root in roots:
        stack.append(root)

    current_path.append(None)
    while len(stack) > 0:
        current = stack[-1]
        if current == current_path[-1]:
            current_path.pop()
            stack.pop()
            current_visited.remove(current)
            block_list.add(current)
            continue
        else:
            current_path.append(current)
            current_visited.add(current)

        children = outs.get(current, [])
        for child in children:
            if child.target in current_visited:
                return True
            elif child.target not in block_list:
                stack.append(child.target)

    return False
