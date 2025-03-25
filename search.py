import sys
from collections import deque, defaultdict

# Function to parse the input file and construct the graph, origin, and destinations
def parse_input_file(filename):
    # Open and read all non-empty lines from the input file
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    graph = defaultdict(list)  # Graph stored as an adjacency list
    nodes = {}                # Stores coordinates of nodes (not used in BFS logic)
    edges = {}                # Not used directly here
    origin = None             # Origin node
    destinations = set()      # Set of destination node(s)
    
    section = None  # To track the current section of the file being parsed

    # Loop through all lines to build graph components
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
            destinations = set(map(int, line.strip().split(";")))

    # Return the constructed graph and start/end info
    return graph, origin, destinations

# Breadth-First Search (BFS) implementation
def bfs(graph, origin, destinations):
    visited = set()                 # Keep track of visited nodes
    queue = deque()                # FIFO queue for BFS
    queue.append((origin, [origin]))  # Start from origin with path containing just origin
    visited.add(origin)
    nodes_created = 1              # Counter for number of nodes added to queue

    # Start BFS loop
    while queue:
        current_node, path = queue.popleft()  # Pop the next node from the queue

        # Check if current node is one of the destinations
        if current_node in destinations:
            return current_node, nodes_created, path

        # Get all neighbors, sorted in ascending order by node number
        neighbors = sorted(graph[current_node], key=lambda x: x[0])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark as visited
                queue.append((neighbor, path + [neighbor]))  # Append new path
                nodes_created += 1     # Increment created node count

    # If no path is found
    return None, nodes_created, []

# Main function to handle command-line arguments and execute the search
def main():
    # Ensure correct usage with 2 command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]       # First argument = input file
    method = sys.argv[2].upper()  # Second argument = method (e.g., BFS)

    # Check if method is BFS (only BFS is supported in this version)
    if method != "BFS":
        print("Only BFS method is implemented in this version.")
        sys.exit(1)

    # Parse the input file
    graph, origin, destinations = parse_input_file(filename)

    # Run BFS algorithm
    goal, nodes_created, path = bfs(graph, origin, destinations)

    # Print results in the required output format
    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {nodes_created}")
        print("->".join(map(str, path)))  # Print path in arrow format
    else:
        print("No path found.")  # No path to any destination

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
