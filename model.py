import math

class Node:
    def __init__(self, ID, x, y, demand):
        self.ID = ID
        self.x = x
        self.y = y
        self.demand = demand

    
nodes = []


with open('data.txt', 'r') as file:  
    lines = file.readlines()
    for line in lines[5:]:  
        parts = line.strip().split(',')
        ID = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        demand = float(parts[3])
        nodes.append(Node(ID, x, y, demand))

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


class Route:
    def __init__(self, dp, cap):
        self.sequenceOfNodes = []
        self.sequenceOfNodes.append(dp)
        self.sequenceOfNodes.append(dp)
        self.cost = 0
        self.capacity = cap
        self.load = 0

