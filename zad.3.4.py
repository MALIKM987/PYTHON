from collections import defaultdict, deque


class AhoCorasick:
    def __init__(self):
        self.trie = {}
        self.output = defaultdict(list)
        self.fail = {}

    def build_trie(self, patterns):
  
        for pattern in patterns:
            node = self.trie
            for char in pattern:
                if char not in node:
                    node[char] = {}
                node = node[char]
            self.output[id(node)].append(pattern)

    def build_failure_links(self):

        queue = deque()
        for char, node in self.trie.items():
            queue.append((node, char))
            self.fail[id(node)] = self.trie

        while queue:
            current_node, current_char = queue.popleft()

            for char, child in current_node.items():
                queue.append((child, char))
                fail_state = self.fail[id(current_node)]
                while fail_state is not None and char not in fail_state:
                    fail_state = self.fail.get(id(fail_state))
                self.fail[id(child)] = fail_state[char] if fail_state else self.trie
                self.output[id(child)] += self.output.get(id(self.fail[id(child)]), [])

    def search(self, text):

        node = self.trie
        matches = []
        for i, char in enumerate(text):
            while node is not None and char not in node:
                node = self.fail.get(id(node))
            if node is None:
                node = self.trie
                continue
            node = node[char]
            for pattern in self.output[id(node)]:
                matches.append((i - len(pattern) + 1, pattern))
        return matches


patterns = ["he", "she", "his", "hers"]
text = "ushers"

ac = AhoCorasick()
ac.build_trie(patterns)
ac.build_failure_links()
matches = ac.search(text)

print("Znalezione wzorce:")
for index, pattern in matches:
    print(f"Wzorzec '{pattern}' znaleziony na pozycji {index}.")
