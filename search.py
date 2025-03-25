import sys
from collections import deque, defaultdict

def parse_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    graph = defaultdict(list)
    nodes = {}
    edges = {}
    origin = None
    destinations = set()
    
    section = None
    for line in lines:
        if line.startswith("Nodes:"):
            section = "nodes"
        elif line.startswith("Edges:"):
            section = "edges"
        elif line.startswith("Origin:"):
            section = "origin"
        elif line.startswith("Destinations:"):
            section = "destinations"
        elif section == "nodes":
            node_id, coord = line.split(":")
            node_id = int(node_id.strip())
            nodes[node_id] = tuple(map(int, coord.strip().strip("()").split(",")))
        elif section == "edges":
            edge_part, cost = line.split(":")
            n1, n2 = map(int, edge_part.strip("()").split(","))
            graph[n1].append((n2, int(cost.strip())))
        elif section == "origin":
            origin = int(line.strip())
        elif section == "destinations":
            destinations = set(map(int, line.strip().split(";")))

    return graph, origin, destinations

def bfs(graph, origin, destinations):
    visited = set()
    queue = deque()
    queue.append((origin, [origin]))
    visited.add(origin)
    nodes_created = 1

    while queue:
        current_node, path = queue.popleft()
        if current_node in destinations:
            return current_node, nodes_created, path

        neighbors = sorted(graph[current_node], key=lambda x: x[0])  # sort by node number
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                nodes_created += 1

    return None, nodes_created, []

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method != "BFS":
        print("Only BFS method is implemented in this version.")
        sys.exit(1)

    graph, origin, destinations = parse_input_file(filename)
    goal, nodes_created, path = bfs(graph, origin, destinations)

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {nodes_created}")
        print("->".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
