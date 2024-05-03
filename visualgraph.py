from graphviz import Digraph
from autodiff.engine import Value


def trace(root):
    nodes, edges = set(), set()
    # def build(v):
    #     if v not in nodes:
    #         nodes.add(v)
    #         for child in v._prev:
    #             edges.add((child, v))
    #             build(child)
    # build(root)
    # return nodes, edges
    def build(v):
        if isinstance(v, list):  # Check if v is a list
            for elem in v:
                if elem not in nodes:
                    nodes.add(elem)
                    for child in elem._prev:
                        edges.add((child, elem))
                        build(child)
        else:
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges
def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))   
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)   
    return dot

