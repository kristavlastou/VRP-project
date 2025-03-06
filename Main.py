from Solver import *

m = Model()
m.BuildModel()
s = Solver(m)
sol = s.solve()
write_to_file(sol, 'output.txt')




