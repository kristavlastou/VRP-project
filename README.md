# VRP Solver

This is a group project that provides a solution for the **Vehicle Routing Problem (VRP)** by minimizing **ton-kilometers** , using Local Search algorithms. The program reads an input file containing coordinates and demand values, computes the optimal routes, and stores the results in an output file.

---

## Features
- Reads input from `Instance.txt`
- Solves the **VRP** by optimizing routes to minimize **ton-kilometers**
- Outputs the solution to `output.txt`

---

## Installation & Usage

### Clone the repository
```sh
git clone <https://github.com/kristavlastou/VRP-project.git>
cd <https://github.com/kristavlastou/VRP-project.git>
```

### Run the solver
```sh
python main.py
```

---

## Input Format (`Instance.txt`)
The input file should contain a list of locations with their respective coordinates and demand values. The format is:

```txt
CAPACITY,<x>
EMPTY_VEHICLE_WEIGHT,<y>
CUSTOMERS,<z>
NODES INFO
ID,XCOORD,YCOORD,DEMAND
<ID> <X_COORDINATE> <Y_COORDINATE> <DEMAND>
```

Example:

```txt
CAPACITY,10
EMPTY_VEHICLE_WEIGHT,10
CUSTOMERS,200
NODES INFO
ID,XCOORD,YCOORD,DEMAND
1 34.05 -118.25 10
2 36.16 -115.15 15
3 40.71 -74.00 20
```

---

## Output Format (`output.txt`)
The program writes the computed VRP solution to `output.txt`, which includes the optimized routes and total ton-kilometers.

---



## License
This project is licensed under the **MIT License**.

