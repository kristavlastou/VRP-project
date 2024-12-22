import math

class Node:
    def __init__(self, node_id, x, y, demand):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.demand = demand

    
nodes = []


with open('data.txt', 'r') as file:  
    lines = file.readlines()
    for line in lines[5:]:  
        parts = line.strip().split(',')
        node_id = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        demand = float(parts[3])
        nodes.append(Node(node_id, x, y, demand))

def distance(from_node, to_node):
    dx = from_node.x - to_node.x
    dy = from_node.y - to_node.y
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return dist

def calculate_distance_matrix(nodes):
    n = len(nodes)
    distance_matrix = [[0] * n for _ in range(n)]  
    
    for i in range(n):
        for j in range(n):
            if i != j:  
                distance_matrix[i][j] = distance(nodes[i], nodes[j])
    
    return distance_matrix

distance_matrix = calculate_distance_matrix(nodes)

