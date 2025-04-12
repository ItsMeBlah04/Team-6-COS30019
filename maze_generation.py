import random

class MazeGeneration:
    def __init__(self, width, height, filename="generated_maze.txt"):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.filename = filename
        self.maze = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.node_id_map = {}
        self.origin = None
        self.destination = None

    def _in_bounds(self, y, x):
        return 0 < y < self.height and 0 < x < self.width

    def generate(self):
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        dir_to_wall = {
            (-2, 0): (-1, 0),
            (2, 0): (1, 0),
            (0, -2): (0, -1),
            (0, 2): (0, 1)
        }

        cells = [(y, x) for y in range(1, self.height, 2) for x in range(1, self.width, 2)]
        visited = set()
        start = random.choice(cells)
        visited.add(start)
        self.maze[start[0]][start[1]] = 1

        while len(visited) < len(cells):
            cell = random.choice([c for c in cells if c not in visited])
            path = [cell]
            current = cell
            visited_in_walk = {current: 0}

            while current not in visited:
                neighbors = [(current[0] + dy, current[1] + dx)
                             for dy, dx in directions if self._in_bounds(current[0] + dy, current[1] + dx)]
                next_cell = random.choice(neighbors)

                if next_cell in path:
                    loop_start = visited_in_walk[next_cell]
                    path = path[:loop_start + 1]
                    visited_in_walk = {cell: i for i, cell in enumerate(path)}
                else:
                    path.append(next_cell)
                    visited_in_walk[next_cell] = len(path) - 1

                current = next_cell

            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                self.maze[y1][x1] = 1
                wall_y = y1 + dir_to_wall[(y2 - y1, x2 - x1)][0]
                wall_x = x1 + dir_to_wall[(y2 - y1, x2 - x1)][1]
                self.maze[wall_y][wall_x] = 1
            self.maze[path[-1][0]][path[-1][1]] = 1
            visited.update(path)

        self.maze[1][0] = 1
        self.maze[self.height - 2][self.width - 1] = 1

    def export(self):
        node_counter = 1
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1:
                    self.node_id_map[(y, x)] = node_counter
                    node_counter += 1

        edges = []
        for (y, x), node_id in self.node_id_map.items():
            for dy, dx in [(0, 1), (1, 0)]:  
                ny, nx = y + dy, x + dx
                if (ny, nx) in self.node_id_map:
                    neighbor_id = self.node_id_map[(ny, nx)]
                    edges.append(((node_id, neighbor_id), 1))

        self.origin = self.node_id_map.get((1, 0))
        self.destination = self.node_id_map.get((self.height - 2, self.width - 1))

        with open(self.filename, "w") as f:
            f.write("Nodes:\n")
            for (y, x), node_id in self.node_id_map.items():
                f.write(f"{node_id}: ({y},{x})\n")

            f.write("Edges:\n")
            for (node1, node2), weight in edges:
                f.write(f"({node1},{node2}): {weight}\n")

            f.write("Origin:\n")
            f.write(f"{self.origin}\n")

            f.write("Destinations:\n")
            f.write(f"{self.destination}\n")

if __name__ == "__main__":
    maze_gen = MazeGeneration(width=15, height=15, filename="test5.txt")
    maze_gen.generate()
    maze_gen.export()