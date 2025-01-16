import math


class Model:

    def __init__(self):
        self.allNodes = []
        self.customers = []
        self.matrix = []
        self.capacity = -1
        

    def BuildModel(self):
        
        with open('Instance.txt', 'r') as file:  
         lines = file.readlines()
         for line in lines[5:]:  
            parts = line.strip().split(',')
            ID = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            demand = float(parts[3])
            self.allNodes.append(Node(ID, x, y, demand))
         depot = self.allNodes[0]
         self.capacity = 10
         totalCustomers = len(self.allNodes) - 1
         self.customers = self.allNodes[1:]
        
         self.matrix = self.calculate_distance_matrix()
        
        #print("All nodes:") #debug
        #for node in self.allNodes:
                #print(f"Node ID: {node.ID}, x: {node.x}, y: {node.y}, demand: {node.demand}")

    def distance(self,from_node, to_node):
        dx = from_node.x - to_node.x
        dy = from_node.y - to_node.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        return dist
    
    def calculate_distance_matrix(self):
        n = len(self.allNodes)  
        distance_matrix = [[0] * n for _ in range(n)]  
        
       
        for i in range(n):
            for j in range(n):
                if i != j:  
                    distance_matrix[i][j] = self.distance(self.allNodes[i], self.allNodes[j])

        return distance_matrix         
    

class Node:
    def __init__(self, ID, x, y, demand):
        self.ID = ID
        self.x = x
        self.y = y
        self.demand = demand
        self.isRouted = False


class Route:
    def __init__(self, dp, cap):
        self.sequenceOfNodes = []
        self.sequenceOfNodes.append(dp)
        self.sequenceOfNodes.append(dp) 
        self.cost = 0
        self.capacity = cap
        self.load = 0



