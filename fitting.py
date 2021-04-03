import numpy as np

class SliceFit:
    def __init__(self, t, strikes, volData, fitter, constraints):
        """
        I assume that 
        1) the voldata is xx.xxx types, e.g., 25.22 which menas 0.2522
        2) strikes is not log
        """
        self.t = t
        self.volData = np.numpy(volData)
        self.totlaVariance = (volData*0.01)**2.0 * t
        self.logStrikes = np.log(strikes)
        self.fitter = fitter
        self.constraints = constraints
        
    def cost_function(self, x):
        """
        cost_function(self, x)
        rho, theta, eta, gamma = x
        """
        self.fitter.reset(x)
        vf = np.vectorize(self.fitter)
        _value = np.sum( (self.totlaVariance - vf(self.logStrikes))**2.0 )
        return _value
        

