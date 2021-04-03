import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ssvi
import phi

class SliceFit:
    def __init__(self, t, strikes, volData,
                 fitter = ssvi.Ssvi([-0.01, 0.01], phi.QuotientPhi([0.01, 0.01])),
                 constraints = ssvi.ssviQuotientConstraints):
        """
        I assume that 
        1) the voldata is xx.xxx types, e.g., 25.22 which menas 0.2522
        2) strikes is not log
        """
        self.t = t
        self.volData = np.array(volData) * 0.01
        self.totalVariance = self.volData**2.0 * t
        self.logStrikes = np.log(strikes)
        self.fitter = fitter
        self.vectorized_fitter = np.vectorize(fitter)
        self.constraints = constraints
        
    def cost_function(self, x):
        """
        cost_function(self, x)
        rho, theta, eta, gamma = x
        or
        rho, theta, lambda = x
        """
        self.fitter.reset(x)
        vf = np.vectorize(self.fitter)
        _value = np.sum( (self.totalVariance - vf(self.logStrikes))**2.0 ) 
        #_value = ( (self.totalVariance /vf(self.logStrikes)  -1.0 )**2.0 ).mean()
        return _value

    def calibrate(self, init = np.array([-0.01, 0.01, 0.01, 0.01]),
                  method='SLSQP', maxiter = 10000, tol = 1.0e-16,
                  verbose = False):
        _init = init
        #import pdb;pdb.set_trace()
        if self.constraints in [{}, None]:
            res = minimize(
                self.cost_function, _init,
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
        else:
            res = minimize(
                self.cost_function, _init,
                constraints = self.constraints,
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
        self.fitter.reset(res.x)
        self.params = res.x
        self.vectorized_fitter = np.vectorize(self.fitter)

    def fitted_vol(self, k):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt(self.vectorized_fitter(k) /self.t  )

    def fitted_slice(self):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt( self.vectorized_fitter(self.logStrikes) / self.t )

    def visualize(self, s=f''):
        st = np.exp(self.logStrikes)
        plt.plot(st, self.fitted_slice(), 'r--')
        plt.plot(st, self.volData, 'b^', label = s+'T ='+"{:10.2f}".format(self.t))
        plt.legend()
        plt.show()
        
        
        

