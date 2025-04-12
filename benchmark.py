import os
import matplotlib.pyplot as plt
import numpy as np

from search import parse_input_file, bfs, dfs, gbfs, AS, cus1, cus2, calculate_path_cost

class Benchmark():
    def __init__(self, algorithms=None):
        self.algorithms = algorithms if algorithms else {
            "BFS": bfs,
            "DFS": dfs,
            "GBFS": gbfs,
            "AStar": AS,
            "Custom 1": cus1,
            "Custom 2": cus2
        }

    def run(self):
        main_folder = os.path.dirname(os.path.abspath(__file__))

        algorithm_names = list(self.algorithms.keys())
        nodes_created_avg = []
        path_cost_avg = []

        for algorithm_name, algorithm in self.algorithms.items():
            nodes_total = []
            path_cost_total = []

            for file in os.listdir(main_folder):
                if file.endswith(".txt"):
                    file_path = os.path.join(main_folder, file)
                    graph, nodes, origin, goal = parse_input_file(file_path)

                    if algorithm_name in ["GBFS", "AStar", "Custom 2"]:
                        destination, nodes_created, path = algorithm(graph, origin, goal, nodes)
                    else:
                        destination, nodes_created, path = algorithm(graph, origin, goal)

                    if destination:
                        cost = calculate_path_cost(graph, path)
                    else:
                        nodes_created = 20
                        cost = 20

                        if file == "test2.txt":
                            cost = 90
                            nodes_created = 60

                    nodes_total.append(nodes_created)
                    path_cost_total.append(cost)

            # Average across all test files
            nodes_created_avg.append(np.mean(nodes_total))
            path_cost_avg.append(np.mean(path_cost_total))

        x_alg = np.arange(len(algorithm_names))
        width = 0.35

        # === 1. Average Summary (Combined) ===
        plt.figure(figsize=(12, 8))
        plt.bar(x_alg - width/2, nodes_created_avg, width, label='Avg. Nodes Created', color='steelblue')
        plt.bar(x_alg + width/2, path_cost_avg, width, label='Avg. Path Cost', color='orange')
        plt.xticks(x_alg, algorithm_names, rotation=45)
        plt.ylabel("Average Values")
        plt.xlabel("Algorithms")
        plt.title("Benchmarking Search Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("benchmark_results.png")

        # === 2. Combined Path Cost per Testcase for All Algorithms ===
        filenames = sorted([file for file in os.listdir(main_folder) if file.endswith(".txt")])
        x = np.arange(len(filenames))

        plt.figure(figsize=(14, 7))
        for algorithm_name, algorithm in self.algorithms.items():
            path_costs = []

            for file in filenames:
                file_path = os.path.join(main_folder, file)
                graph, nodes, origin, goal = parse_input_file(file_path)

                if algorithm_name in ["GBFS", "AStar", "Custom 2"]:
                    destination, nodes_created_temp, path = algorithm(graph, origin, goal, nodes)
                else:
                    destination, nodes_created_temp, path = algorithm(graph, origin, goal)

                cost = calculate_path_cost(graph, path) if destination else (90 if file == "test2.txt" else 20)
                path_costs.append(cost)

            plt.plot(x, path_costs, marker='o', label=algorithm_name)

        plt.xticks(x, filenames, rotation=0)
        plt.xlabel("Testcase")
        plt.ylabel("Path Cost")
        plt.title("Path Cost per Testcase for All Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("path_cost_per_testcase.png")


        # === 3. Combined Nodes Created per Testcase for All Algorithms ===
        plt.figure(figsize=(14, 7))
        for algorithm_name, algorithm in self.algorithms.items():
            nodes_created_list = []

            for file in filenames:
                file_path = os.path.join(main_folder, file)
                graph, nodes, origin, goal = parse_input_file(file_path)

                if algorithm_name in ["GBFS", "AStar", "Custom 2"]:
                    destination, nodes_created_temp, path = algorithm(graph, origin, goal, nodes)
                else:
                    destination, nodes_created_temp, path = algorithm(graph, origin, goal)

                if not destination:
                    nodes_created_temp = 60 if file == "test2.txt" else 20

                nodes_created_list.append(nodes_created_temp)

            plt.plot(x, nodes_created_list, marker='o', label=algorithm_name)

        plt.xticks(x, filenames, rotation=0)
        plt.xlabel("Testcase Filename")
        plt.ylabel("Nodes Created")
        plt.title("Nodes Created per Testcase for All Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("nodes_created_per_testcase.png")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()