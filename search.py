import sys
from collections import deque, defaultdict
from heapq import heappush, heappop
from math import sqrt

# Parse input file
def parse_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    graph = defaultdict(list)
    nodes = {}
    origin = None
    goals = set()
    
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
            goals = set(map(int, line.strip().split(";")))
    
    return graph, nodes, origin, goals

# Breadth-First Search
def bfs(graph, origin, goals):
    visited = set()
    queue = deque()
    queue.append((origin, [origin]))
    visited.add(origin)
    nodes_created = 1
    while queue:
        current_node, path = queue.popleft()
        if current_node in goals:
            return current_node, nodes_created, path
        neighbors = sorted(graph[current_node], key=lambda x: x[0])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                nodes_created += 1
    return None, nodes_created, []

# Depth-First Search
def dfs(graph, origin, goals):
    visited = set()
    stack = []
    stack.append((origin, [origin]))
    visited.add(origin)
    nodes_created = 1
    while stack:
        current_node, path = stack.pop()
        if current_node in goals:
            return current_node, nodes_created, path
        neighbors = sorted(graph[current_node], key=lambda x: x[0])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
                nodes_created += 1
    return None, nodes_created, []

# A* Search
def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def AS(graph, origin, goals, coords):
    def heuristic(node):
        return min(euclidean_distance(*coords[node], *coords[g]) for g in goals)
    
    h_start = heuristic(origin)
    pq = [(h_start, 0, origin, [origin])]  # (f, g, node, path)
    visited = set()
    nodes_created = 1
    
    while pq:
        f, g, node, path = heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node in goals:
            return node, nodes_created, path
        for neighbor, cost in graph.get(node, []):
            if neighbor not in visited:
                new_g = g + cost
                h = heuristic(neighbor)
                new_f = new_g + h
                heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
                nodes_created += 1
    return None, nodes_created, []

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    methods = ["BFS", "DFS", "AS"]

    if method not in methods:
        print("Unsupported search method.\nSupported arguments are:")
        print(*methods, sep=', ')
        sys.exit(1)

    graph, nodes, origin, goals = parse_input_file(filename)

    if method == "BFS":
        goal, nodes_created, path = bfs(graph, origin, goals)
    elif method == "DFS":
        goal, nodes_created, path = dfs(graph, origin, goals)
    elif method == "AS":
        goal, nodes_created, path = AS(graph, origin, goals, nodes)

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {nodes_created}")
        print("->".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
