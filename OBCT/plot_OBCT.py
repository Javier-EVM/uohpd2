import matplotlib.pyplot as plt
import networkx as nx

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Función para construir un árbol binario con una profundidad específica
def build_binary_tree(depth, current_depth=1):
    if current_depth > depth:
        return None
    root = TreeNode(f"Node {current_depth}")
    root.left = build_binary_tree(depth, current_depth + 1)
    root.right = build_binary_tree(depth, current_depth + 1)
    return root

# Función para plotear un árbol binario con etiquetas personalizadas en cada nodo
def plot_binary_tree(root):
    G = nx.Graph()

    def add_edges(node, parent=None):
        if node:
            node_label = node.value
            G.add_node(node_label)
            if parent:
                G.add_edge(parent, node_label)
            add_edges(node.left, node_label)
            add_edges(node.right, node_label)

    add_edges(root)

    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # Diseño tipo árbol

    labels = {node: node for node in G.nodes()}

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=5000, node_color='skyblue', font_size=10)
    plt.axis('off')
    plt.show()

# Ejemplo de uso: Crear un árbol binario con profundidad 3
if __name__ == "__main__":
    depth = 3
    root = build_binary_tree(depth)
    plot_binary_tree(root)
