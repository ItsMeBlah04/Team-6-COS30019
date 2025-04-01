import sys
from collections import deque, defaultdict
from heapq import heappush, heappop
from math import sqrt

# Function to parse the input file and construct the graph, origin, and destinations
def parse_input_file(filename):
    # Open and read all non-empty lines from the input file
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    graph = defaultdict(list) # Graph stored as an adjacency list
    nodes = {}  # Stores coordinates of nodes (not used in BFS logic)
    origin = None # Origin node
    goals = set() # To track the current section of the file being parsed
    
    section = None
    for line in lines: # Loop through all lines to build graph components
        if line.startswith("Nodes:"):
            section = "nodes"
        elif line.startswith("Edges:"):
            section = "edges"
        elif line.startswith("Origin:"):
            section = "origin"
        elif line.startswith("Destinations:"):
            section = "destinations"
        elif section == "nodes":
            # Parse node and its coordinate
            node_id, coord = line.split(":")
            node_id = int(node_id.strip())
            nodes[node_id] = tuple(map(int, coord.strip().strip("()").split(",")))
        elif section == "edges":
            # Parse edge (directed) and cost, add to graph
            edge_part, cost = line.split(":")
            n1, n2 = map(int, edge_part.strip("()").split(","))
            graph[n1].append((n2, int(cost.strip())))
        elif section == "origin":
            # Parse origin node
            origin = int(line.strip())
        elif section == "destinations":
            # Parse multiple destination nodes
            goals = set(map(int, line.strip().split(";")))
    # Return the constructed graph and start/end info
    return graph, nodes, origin, goals

# Breadth-First Search (BFS) implementation
def bfs(graph, origin, goals):
    visited = set() # Keep track of visited nodes
    queue = deque() # FIFO queue for BFS
    queue.append((origin, [origin])) # Start from origin with path containing just origin
    visited.add(origin)
    nodes_created = 1 # Counter for number of nodes added to queue
    while queue:
        current_node, path = queue.popleft() # Pop the next node from the queue

        # Check if current node is one of the destinations
        if current_node in goals:
            return current_node, nodes_created, path
        # Get all neighbors, sorted in ascending order by node number
        neighbors = sorted(graph[current_node], key=lambda x: x[0])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor) # Mark as visited
                queue.append((neighbor, path + [neighbor])) # Append new path
                nodes_created += 1 # Increment created node count

    # If no path is found            
    return None, nodes_created, []

# Main function to handle command-line arguments and execute the search
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

# A* (revamped)
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

# Main
def main():
        # Ensure correct usage with 2 command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1] # First argument = input file
    method = sys.argv[2].upper() # Second argument = method (e.g., BFS)
    methods = ["BFS", "DFS", "AS"]
    # Check if method is BFS (only BFS is supported in this version)
    if method not in methods:
        print("Unsupported search method.\nSupported arguments are:")
        print(*methods, sep=', ')
        sys.exit(1)
    # Parse the input file
    graph, nodes, origin, goals = parse_input_file(filename)
    # Run BFS algorithm
    if method == "BFS":
        goal, nodes_created, path = bfs(graph, origin, goals)
    elif method == "DFS":
        goal, nodes_created, path = dfs(graph, origin, goals)
    elif method == "AS":
        goal, nodes_created, path = AS(graph, origin, goals, nodes)
    # Print results in the required output format
    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {nodes_created}") 
        print("->".join(map(str, path))) # Print path in arrow format
    else:
        print("No path found.") # No path to any destination
# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
