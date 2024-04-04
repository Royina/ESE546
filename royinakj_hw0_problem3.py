#!/usr/bin/env python
# coding: utf-8

# In[22]:


from scipy.optimize import minimize
import math
import numpy as np

fun = lambda x : x[0]**2 + x[1]**2 - 6*x[0]*x[1] - 4*x[0] - 5*x[1]
fun_deriv = lambda x : np.array([2*x[0]-6*x[1]-4,2*x[1]-6*x[0]-5])

cons = ({'type': 'ineq', 'fun': lambda x :  np.array(x[1] + (x[0] - 2)**2 - 4),'jac':lambda x: np.array([2*(x[0]-2), 1])},
        {'type': 'ineq', 'fun': lambda x : - x[0] - x[1] + 1, 'jac': np.array([-1, -1])})

bnds = ((- math.inf, 0), (- math.inf, 0))

res = minimize(fun, (0.4, 0.5), method='Powell', constraints=cons, bounds = bnds, jac=fun_deriv)

res.x

