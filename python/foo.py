from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor

from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed
# example objective taken from skopt
from skopt.benchmarks import branin

optimizer = Optimizer(
    base_estimator=GaussianProcessRegressor(),
    dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)]
)

for i in range(10): 
    x = optimizer.ask(n_points=4)  # x is a list of n_points points    
    y = Parallel(n_jobs=2)(delayed(branin)(v) for v in x)  # evaluate points in parallel
    optimizer.tell(x, y)

# takes ~ 20 sec to get here
print(min(optimizer.yi))  # print the best objective found 