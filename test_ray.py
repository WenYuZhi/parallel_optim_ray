import ray
import gurobipy as gp
from gurobipy import GRB

@ray.remote
class TestRay():
    def __init__(self, c, n_dim) -> None:
        self.c, self.n_dim = c, n_dim
        self.m = gp.Model()
        self.x = self.m.addVars(self.n_dim, self.n_dim, lb = 0, ub = 1.0, vtype = GRB.CONTINUOUS)
    
    def add_constrs(self):
        self.m.addConstrs(gp.quicksum(self.x[i,j] for i in range(4)) == 1 for j in range(4))
        self.m.addConstrs(gp.quicksum(self.x[i,j] for j in range(4)) == 1 for i in range(4))
    
    def set_objective(self):
        self.m.setObjective(gp.quicksum(self.c[i,j]*self.x[i,j] for i in range(self.n_dim) for j in range(self.n_dim)))

    def optimize(self):
        self.m.optimize()
        return self.m.ObjVal

class BaseLine():
    def __init__(self, c, n_dim) -> None:
        self.c, self.n_dim = c, n_dim
        self.m = gp.Model()
        self.x = self.m.addVars(self.n_dim, self.n_dim, vtype = GRB.BINARY)
    
    def add_constrs(self):
        self.m.addConstrs(gp.quicksum(self.x[i,j] for i in range(4)) == 1 for j in range(4))
        self.m.addConstrs(gp.quicksum(self.x[i,j] for j in range(4)) == 1 for i in range(4))
    
    def set_objective(self):
        self.m.setObjective(gp.quicksum(self.c[i,j]*self.x[i,j] for i in range(self.n_dim) for j in range(self.n_dim)))

    def optimize(self):
        self.m.optimize()
        return self.m.ObjVal
