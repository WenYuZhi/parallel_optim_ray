import ray
import numpy as np
import datetime 
from test_ray import TestRay, BaseLine
import pandas as pd

LOG_FILE_PATH = './log/{}'

test_ray, n_dim, n_problem = [], 50, 10
cost_coeff = [np.random.random((n_dim, n_dim)) for i in range(n_problem)]

for i in range(n_problem):
    test_ray.append(TestRay.remote(cost_coeff[i], n_dim))
    test_ray[i].add_constrs.remote()
    test_ray[i].set_objective.remote()

results_ray = []
ts = datetime.datetime.now()
for i in range(n_problem):
    obj = ray.get(test_ray[i].optimize.remote())
    results_ray.append(obj)

cpu_time1 = (datetime.datetime.now() - ts).seconds

base_line = []
for i in range(n_problem):
    base_line.append(BaseLine(cost_coeff[i], n_dim))
    base_line[i].add_constrs()
    base_line[i].set_objective()

results_base_line = []
ts = datetime.datetime.now()
for i in range(n_problem):
    obj = base_line[i].optimize()
    results_base_line.append(obj)

cpu_time2 = (datetime.datetime.now() - ts).seconds

log_file = pd.DataFrame([results_base_line, results_ray]).T
log_file.columns = ['without parallel', 'with parallel']
log_file.to_csv(LOG_FILE_PATH.format('obj_vals_nd_{}_np_{}.csv'.format(n_dim, n_problem)))

log_file = pd.DataFrame([cpu_time1, cpu_time2]).T
log_file.columns = ['without parallel', 'with parallel']
log_file.to_csv(LOG_FILE_PATH.format('cpu_time_nd_{}_np_{}.csv'.format(n_dim, n_problem)))

