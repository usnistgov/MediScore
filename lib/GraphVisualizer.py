import os
import pydot
import cv2

def generate_thumbnail(input_fn, output_fn, height = 64):
    # Resize to fixed height while preserving aspect ratio
    if not os.path.isfile(output_fn):
        input_img = cv2.imread(input_fn)

        scale = float(height) / input_img.shape[0]
        out_dims = (int(input_img.shape[1] * scale), height)

        output_img = cv2.resize(input_img, out_dims)
        cv2.imwrite(output_fn, output_img)

# For the nodes argument, expecting an enumerable of tuples of the
# form (id, attrs) where attrs is a dict of pydot attributes for the
# node.  For the edges argument, expecting an enumerable of tuples of
# the form (src_id, dest_id, attrs).
def render_provenance_graph(nodes, edges, output_fn):
    graph = pydot.Dot(graph_type='digraph')

    node_default_attr = { "shape": "rectangle" }
    edge_default_attr = { "shape": "plain" }

    def _node_to_pydot(node_id, attrs={}):
        new_attrs = node_default_attr.copy()
        new_attrs.update(attrs)

        return pydot.Node(node_id, **new_attrs)

    def _edge_to_pydot(src_id, dest_id, attrs={}):
        new_attrs = edge_default_attr.copy()
        new_attrs.update(attrs)

        return pydot.Edge(src_id, dest_id, **new_attrs)

    for node in nodes:
        graph.add_node(_node_to_pydot(*node))

    for edge in edges:
        graph.add_edge(_edge_to_pydot(*edge))

    graph.write_png(output_fn)

def render_provenance_graph_from_mapping(correct_nodes, fa_nodes, missing_nodes, correct_edges, fa_edges, missing_edges, output_fn, ref_dir = None):
    correct_color = "green"
    fa_color = "red"
    missing_color = "dimgray"

    thumb_dir = os.path.join(os.path.dirname(output_fn), "thumbs")
    if not os.path.isdir(thumb_dir):
        os.makedirs(thumb_dir)
    
    def _generate_label(node_id):
        node_name = os.path.basename(node_id)
        basename, ext = os.path.splitext(node_name)

        if ref_dir is not None:
            img_path = os.path.join(ref_dir, node_id)
            
            if os.path.isfile(img_path):
                # Generate thumbnail
                output_thumb_fn = os.path.join(thumb_dir, "{}_thumb{}".format(basename, ".jpg"))
                generate_thumbnail(img_path, output_thumb_fn)
                
                return "<<TABLE border=\"0\" cellborder=\"0\"><TR><TD><IMG src=\"{}\"/></TD></TR><TR><TD>{}</TD></TR></TABLE>>".format(output_thumb_fn, node_name)

        return node_name        
    
    nodes = ([ (n, { "color": correct_color, "label": _generate_label(n) }) for n in correct_nodes ] +
             [ (n, { "color": fa_color, "label": _generate_label(n) }) for n in fa_nodes ] +
             [ (n, { "color": missing_color, "label": _generate_label(n) }) for n in missing_nodes ])
    
    edges = ([ (s, t, { "color": correct_color }) for s, t in correct_edges ] +
             [ (s, t, { "color": fa_color }) for s, t in fa_edges ] +
             [ (s, t, { "color": missing_color }) for s, t in missing_edges ])

    render_provenance_graph(nodes, edges, output_fn)
    
