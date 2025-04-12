import os
import matplotlib.pyplot as plt
import numpy as np

from search import parse_input_file, bfs, dfs, gbfs, AS, cus1, cus2, calculate_path_cost

class Benchmark:
    """
    A benchmarking class to evaluate and compare different search algorithms 
    based on path cost and number of nodes created. It also provides visualization of results.
    
    Parameters:
    - algorithms (dict): A dictionary of algorithm names mapped to their function implementations.
                        If not provided, a default set of algorithms will be used.
    """
    def __init__(self, algorithms=None):
        self.algorithms = algorithms if algorithms else {
            "BFS": bfs,
            "DFS": dfs,
            "GBFS": gbfs,
            "AStar": AS,
            "Custom 1": cus1,
            "Custom 2": cus2
        }

    def run_algorithm(self, algorithm, algorithm_name, file_path, filename):
        """
        Executes a specified search algorithm on the parsed input file.
        
        Parameters:
        - algorithm (function): The search algorithm function to run.
        - algorithm_name (str): Name of the algorithm (used to check for heuristic-based algorithms).
        - file_path (str): Full path to the input maze/graph file.
        - filename (str): Name of the file (used for display/logging purposes).
        
        Returns:
        - cost (int): Total cost of the path found by the algorithm.
        - nodes_created (int): Number of nodes generated during the search.
        """
        graph, nodes, origin, goal = parse_input_file(file_path)

        if algorithm_name in ["GBFS", "AStar", "Custom 2"]:
            destination, nodes_created, path = algorithm(graph, origin, goal, nodes)
        else:
            destination, nodes_created, path = algorithm(graph, origin, goal)

        cost = calculate_path_cost(graph, path)

        return cost, nodes_created

    def visualize_averages(self, results, algorithm_names):
        """
        Plots a bar chart comparing the average path cost and nodes created for each algorithm.
        
        Parameters:
        - results (dict): Contains the benchmarking data (costs and node counts).
        - algorithm_names (list): List of algorithm names to include in the chart.
        
        Output:
        - Saves a bar chart as 'benchmark_results.png'.
        """
        nodes_created_avg = [np.mean(results[name]["nodes"]) for name in algorithm_names]
        path_cost_avg = [np.mean(results[name]["costs"]) for name in algorithm_names]

        x_alg = np.arange(len(algorithm_names))
        width = 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x_alg - width / 2, nodes_created_avg, width, label='Avg. Nodes Created')
        plt.bar(x_alg + width / 2, path_cost_avg, width, label='Avg. Path Cost')
        plt.xticks(x_alg, algorithm_names, rotation=45)
        plt.ylabel("Average Values")
        plt.xlabel("Algorithms")
        plt.title("Benchmarking Search Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./benchmark/benchmark_results.png")

    def visualize_path_costs(self, results, algorithm_names, filenames):
        """
        Plots a line graph of path costs per test case for each algorithm.
        
        Parameters:
        - results (dict): Benchmark results with path costs.
        - algorithm_names (list): Names of the algorithms to plot.
        - filenames (list): List of test case filenames.
        
        Output:
        - Saves the plot as 'path_cost_per_testcase.png'.
        """
        x = np.arange(len(filenames))
        plt.figure(figsize=(14, 7))

        for name in algorithm_names:
            plt.plot(x, results[name]["costs"], marker='o', label=name)

        plt.xticks(x, filenames, rotation=0)
        plt.xlabel("Testcase")
        plt.ylabel("Path Cost")
        plt.title("Path Cost per Testcase for All Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./benchmark/path_cost_per_testcase.png")

    def visualize_nodes_created(self, results, algorithm_names, filenames):
        """
        Plots a line graph showing how many nodes were created per test case for each algorithm.
        
        Parameters:
        - results (dict): Benchmark results with node creation data.
        - algorithm_names (list): Names of the algorithms to plot.
        - filenames (list): List of test case filenames.
        
        Output:
        - Saves the plot as 'nodes_created_per_testcase.png'.
        """
        x = np.arange(len(filenames))
        plt.figure(figsize=(14, 7))

        for name in algorithm_names:
            plt.plot(x, results[name]["nodes"], marker='o', label=name)

        plt.xticks(x, filenames, rotation=0)
        plt.xlabel("Testcase Filename")
        plt.ylabel("Nodes Created")
        plt.title("Nodes Created per Testcase for All Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./benchmark/nodes_created_per_testcase.png")

    def run(self):
        """
        Runs benchmarking for all available text files in the current directory.
        
        Process:
        - Parses each .txt file in the folder.
        - Executes each algorithm on each file.
        - Stores and aggregates results.
        - Visualizes results as charts.
        
        Output:
        - Saves three visual plots in the 'benchmark' directory.
        """
        main_folder = os.path.dirname(os.path.abspath(__file__))
        filenames = sorted([file for file in os.listdir(main_folder) if file.endswith(".txt")])
        algorithm_names = list(self.algorithms.keys())

        results = {name: {"nodes": [], "costs": []} for name in algorithm_names}

        for filename in filenames:
            file_path = os.path.join(main_folder, filename)

            for name, algorithm in self.algorithms.items():
                cost, nodes = self.run_algorithm(algorithm, name, file_path, filename)
                results[name]["costs"].append(cost)
                results[name]["nodes"].append(nodes)

        self.visualize_averages(results, algorithm_names)
        self.visualize_path_costs(results, algorithm_names, filenames)
        self.visualize_nodes_created(results, algorithm_names, filenames)

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()