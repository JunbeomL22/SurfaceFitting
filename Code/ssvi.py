import numpy as np
import constant

class FittingFunction:
    def __init__(self):
        assert False, "FittingFunction is not implemented"

    def __call__(self, k):
        return 0.0

    def reset(x):
        pass
    
class Ssvi(FittingFunction):
    def __init__(self, x, phi):
        self.rho  = x[0]
        self.theta= x[1]
        self.phi  = phi

    def __call__(self, k):
        r = self.rho
        j = self.theta
        p = self.phi
        ret = 1.0 + r*p(j)*k + np.sqrt( (p(j)*k + r)**2.0 + (1.0 - r**2.0)  )
        ret *= j/2.0
        return ret
    
    def reset(self, x):
        #import pdb;pdb.set_trace()
        self.rho = x[0]
        self.theta = x[1]
        self.phi = type(self.phi)(x[2:])

    
eps = constant.bumpBuffer
ssviQuotientConstraints = [
    # rho
    {'type': 'ineq',
     'fun': lambda x: np.array([1.0 - eps - np.abs(x[0])]),
     'jac': lambda x: np.array([-np.sign(x[0]), 0., 0., 0.])
     },
    # theta
    {'type': 'ineq',
     'fun': lambda x: np.array([x[1] - eps]),
     'jac': lambda x: np.array([0., 1., 0., 0.])
     },
    # eta
    {'type': 'ineq',
     'fun': lambda x: np.array([x[2] - eps]),
     'jac': lambda x: np.array([0., 0., 1., 0.])
     },
    # gamma: gamma >= eps
    {'type': 'ineq',
     'fun': lambda x: np.array([x[3] - eps]),
     'jac': lambda x: np.array([0., 0., 0., 1.])
     },
    # gamma: gamma <= 1 - eps
    {'type': 'ineq',
     'fun': lambda x: np.array([1.0 - eps - x[3]]),
     'jac': lambda x: np.array([0., 0., 0., -1.])
     }, 
    # butterfly (eta + eps) * (1 + (|rho|+eps) ) < 2
    {'type': 'ineq',
     'fun': lambda x: np.array([2.0 - (x[2] + eps) * (1.0 + np.abs(x[0]) + eps) ]),
     'jac': lambda x: np.array([-(x[2] + eps), 0., -(1.0 + np.abs(x[0]) + eps), 0.])
     } 
]
        

    
