class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = {}  # Słownik w postaci {krawędź: dziecko}

    def __str__(self):
        return str(self.value)


class Tree:

    def __init__(self, root_value):
        self.root = TreeNode(root_value)

    def add_edge(self, parent, child_value, edge_value=None):

        child_node = TreeNode(child_value)
        parent.children[edge_value] = child_node
        return child_node

    def traverse(self, node=None, depth=0):

        if node is None:
            node = self.root
        print("  " * depth + f"Node: {node.value}")
        for edge, child in node.children.items():
            print("  " * (depth + 1) + f"Edge: {edge}")
            self.traverse(child, depth + 2)

    def __str__(self):

        result = []

        def _build_str(node, depth):
            result.append("  " * depth + f"Node: {node.value}")
            for edge, child in node.children.items():
                result.append("  " * (depth + 1) + f"Edge: {edge}")
                _build_str(child, depth + 2)

        _build_str(self.root, 0)
        return "\n".join(result)


tree = Tree("Root")
node_a = tree.add_edge(tree.root, "Child A", "Edge 1")
node_b = tree.add_edge(tree.root, "Child B", "Edge 2")
tree.add_edge(node_a, "Child A1", "Edge A1")
tree.add_edge(node_b, "Child B1", "Edge B1")
tree.add_edge(node_b, "Child B2", "Edge B2")

print("Tree Traversal:")
tree.traverse()


print("\nTree Structure:")
print(tree)
