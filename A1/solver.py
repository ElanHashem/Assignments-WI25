import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner
        self.neighbors.append(node)
        node.neighbors.append(self)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    # TODO: Implement the logic to build nodes and link neighbors
    for row in range(len(maze)):
        for column in range(len(maze[row])):
            if row >= 0 and row < len(maze) and column >=0 and column <len(maze[row]) and maze[row][column] == 0:
                nodes_dict[(row,column)] = Node((row,column))
                
    for key in nodes_dict.keys():
        if nodes_dict.get((key[0]+1, key[1])) is not None:
            nodes_dict.get(key).neighbors.append(nodes_dict.get((key[0]+1, key[1])))
        if nodes_dict.get((key[0]-1, key[1])) is not None:
            nodes_dict.get(key).neighbors.append(nodes_dict.get((key[0]-1, key[1])))
        if nodes_dict.get((key[0], key[1]+1)) is not None:
            nodes_dict.get(key).neighbors.append(nodes_dict.get((key[0]+1, key[1]+1)))   
        if nodes_dict.get((key[0], key[1]-1)) is not None:
            nodes_dict.get(key).neighbors.append(nodes_dict.get((key[0]+1, key[1]-1)))
        
    start_node = None
    goal_node = None

    # TODO: Assign start_node and goal_node if they exist in nodes_dict

    if nodes_dict.get((0,0)) is not None:
        start_node = nodes_dict.get((0,0))
    if nodes_dict.get((len(maze)-1,len(maze[row])-1)) is not None:
        goal_node = nodes_dict.get((len(maze)-1,len(maze[row])-1))
    #print(f"{start_node} {goal_node}")
    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    print("This is running")
    # TODO: Implement BFS
    if not start_node or not goal_node:
        print("empty")
        return None  

    queue = deque([start_node])  
    visited = set()             
    parent_map = {}             

    visited.add(start_node)

    while queue:
        current = queue.popleft()

        # Goal check
        if current == goal_node:
            # Reconstruct the path from start to goal
            path = []
            while current:
                path.append(current.value)
                current = parent_map.get(current)
            return path[::-1]  

        # Explore neighbors
        if current is None:
            continue
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                queue.append(neighbor)
        print("looking")
    print("None")
    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    stack = []
    visited = []
    path = []
    def dfs_rec(stack, visited, current_node, goal_node):
        if current_node is None:
            return None
        if current_node == goal_node:
                return stack
        for neighbor in current_node.neighbors:
            if neighbor in visited:
                continue
            stack.append(neighbor)
            visited.append(neighbor)
            test = dfs_rec(stack, visited, neighbor, goal_node)
            if test is not None:
                return test
            else:
                stack.pop()
        return None
        
    nodes =  dfs_rec(stack, visited, start_node, goal_node)
    if nodes is None:
        print("None")
        return None
    path = []
    for node in nodes:
        if node is None:
            continue
        path.append(node.value)
    return path 

###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    
    if not start_node or not goal_node:
        return None  

    # Priority queue (min-heap) for nodes to explore
    open_set = []
    heapq.heappush(open_set, (0, start_node))  
    g_score = {start_node: 0}  # Cost from start_node to current node
    f_score = {start_node: manhattan_distance(start_node, goal_node)}  # Estimated total cost

    parent_map = {}

    while open_set:
        # Get the node with the smallest f_score
        _, current = heapq.heappop(open_set)

        # Goal check
        if current == goal_node:
            # Reconstruct the path
            path = []
            while current:
                path.append(current.value)
                current = parent_map.get(current)
            return path[::-1]  # Reverse the path to get start-to-goal order

        # Explore neighbors
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1  # Assuming all edges have cost 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Update scores
                if neighbor is None:
                    continue
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal_node)

                # Update parent map
                parent_map[neighbor] = current

                # Add to the open set if not already there
                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    x1, y1 = node_a.value
    x2, y2 = node_b.value
    return abs(x1 - x2) + abs(y1 - y2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    if not start_node or not goal_node:
        return None

    # Queues for BFS from start and goal
    start_queue = deque([start_node])
    goal_queue = deque([goal_node])

    # Visited sets
    start_visited = {start_node: None}  # Maps node to its parent
    goal_visited = {goal_node: None}    # Maps node to its parent

    # Helper to reconstruct the path
    def reconstruct_path(meeting_node):
        path_from_start = []
        path_from_goal = []

        # Trace back from meeting_node to start_node
        current = meeting_node
        while current:
            path_from_start.append(current.value)
            current = start_visited[current]

        # Trace back from meeting_node to goal_node
        current = meeting_node
        while current:
            path_from_goal.append(current.value)
            current = goal_visited[current]

        # Combine the paths (reverse goal path and exclude the meeting node)
        return path_from_start[::-1] + path_from_goal[1:]

    # Alternate BFS expansions
    while start_queue and goal_queue:
        # Expand from the start side
        if start_queue:
            current = start_queue.popleft()

            for neighbor in current.neighbors:
                if neighbor not in start_visited:
                    start_visited[neighbor] = current
                    start_queue.append(neighbor)

                    # Check for intersection
                    if neighbor in goal_visited:
                        return reconstruct_path(neighbor)

        # Expand from the goal side
        if goal_queue:
            current = goal_queue.popleft()
            if current is None:
                continue
            for neighbor in current.neighbors:
                if neighbor not in goal_visited:
                    goal_visited[neighbor] = current
                    goal_queue.append(neighbor)

                    # Check for intersection
                    if neighbor in start_visited:
                        return reconstruct_path(neighbor)

    # No path found
    return None

###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    if not start_node or not goal_node:
        return None

    def energy(node):
        """
        Energy function: Returns the heuristic distance from the current node to the goal node.
        """
        return manhattan_distance(node, goal_node)

    def probability(delta_e, temperature):
        """
        Acceptance probability function: Determines whether to accept a worse solution.
        """
        if delta_e < 0:
            return 1.0
        return math.exp(-delta_e / temperature)

    # Initial setup
    current_node = start_node
    best_node = current_node
    best_energy = energy(current_node)
    path = [current_node.value]

    while temperature > min_temperature:
        # Randomly select a neighbor
        if not current_node.neighbors:
            break  # Dead end
        next_node = random.choice(current_node.neighbors)

        if next_node is None:
            continue
        current_energy = energy(current_node)
        next_energy = energy(next_node)
        delta_e = next_energy - current_energy

        if probability(delta_e, temperature) >= random.random():
            current_node = next_node
            path.append(current_node.value)

            if next_energy < best_energy:
                best_node = current_node
                best_energy = next_energy

        # Stop if we reach the goal
        if current_node == goal_node:
            return path

        # Cool down the temperature
        temperature *= cooling_rate

    # If no path is found, return None
    return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    return None


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
