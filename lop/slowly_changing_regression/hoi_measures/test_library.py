from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
from thoi.heuristics import simulated_annealing, greedy
import numpy as np

X = np.random.normal(0,1, (1000, 10))

# Computation of O information for the nplet that consider all the variables of X
measures = nplets_measures(X)

# Computation of O info for a single nplet (it must be a list of nplets even if it is a single nplet)
measures = nplets_measures(X, [[0,1,3]])

# Computation of O info for multiple nplets
measures = nplets_measures(X, [[0,1,3],[3,7,4],[2,6,3]])

# Extensive computation of O information measures over all combinations of features in X
measures = multi_order_measures(X)

# Compute the best 10 combinations of features (nplet) using greedy, starting by exaustive search in 
# lower order and building from there. Result shows best O information for 
# each built optimal orders
best_nplets, best_scores = greedy(X, 3, 5, repeat=10)

# Compute the best 10 combinations of features (nplet) using simulated annealing: There are two initialization options
# 1. Starting by a custom initial solution with shape (repeat, order) explicitely provided by the user.
# 2. Selecting random samples from the order.
# Result shows best O information for each built optimal orders
best_nplets, best_scores = simulated_annealing(X, 5, repeat=10)