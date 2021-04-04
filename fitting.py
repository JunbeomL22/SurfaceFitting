import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ssvi
import phi
from functools import partial
import copy
from datetime import date

class SliceFit:
    """
    I assume that 
    1) the voldata is xx.xxx types, e.g., 25.22 which menas 0.2522
    2) strikes is not log
    
    The variable, weight is Bool type, which determines whether the optimization will consider weights.
    If it is True, the weight will be chosen as 5.0*(abs(log(weight_cut)) - abs(k))^+ for log strike k.
    The default value of weight_cut is 60%. This means, the vol data under 60% is not considered in fitting.
    There are some logics behind this choice. 
    First, for els trading, way deep Put options are not recommded to sell.
    Second, I think of sudden spikes in deep areas as noise.
    Third, high volatility makes trader sell Els by cheaper price. 
    Such cheaper price hinges on the chance to sell deep Put options by high price. 
    But remember why the Put option price is good. 
    This means that a bunch of people want to buy Put option, which is a sign of potential turmoil.
    Therefore, the chain of action, selling Put in good price and selling Els in bad price, 
    can not end well in any way.
    Rather, I want trader not to sell (or passively quote) ELS when the deep volatilities are unusually high.
    """
    def __init__(self, t, strikes, volData,
                 fitter = ssvi.Ssvi([-0.3, 0.01], phi.QuotientPhi([0.4, 0.4])),
                 constraints = ssvi.ssviQuotientConstraints,
                 weight = True,
                 weight_cut = 0.6):
        self.t = t
        self.volData = np.array(volData) * 0.01
        self.totalVariance = self.volData**2.0 * t
        self.logStrikes = np.log(strikes)
        self.fitter = fitter
        self.vectorized_fitter = np.vectorize(fitter)
        self.constraints = constraints
        if weight:
            self.weight = 5.0 * np.maximum(0.0, np.abs(np.log(weight_cut)) - np.abs(self.logStrikes))
        else:
            self.weight = np.repeat(1.0, length(self.logStrikes))
        
    def cost_function(self, x):
        """
        cost_function(self, x)
        rho, theta, eta, gamma = x
        or
        rho, theta, lambda = x
        """
        self.fitter.reset(x)
        vf = np.vectorize(self.fitter)
        _value = np.sum( (self.weight*(self.totalVariance - vf(self.logStrikes)))**2.0 ) 
        #_value = ( (self.totalVariance /vf(self.logStrikes)  -1.0 )**2.0 ).mean()
        return _value

    def calibrate(self, init = np.array([-0.3, 0.01, 0.4, 0.4]),
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

class SurfaceFit:
    """
    volData is now a Matrix
    """
    def __init__(self,
                 calc_date, dates, 
                 strikes, volData,
                 weight = True,
                 weight_cut = 0.6):
        diff = np.array(dates) - calc_date
        self.times = np.array([x.days/365.0 for x in diff])
        self.slice_num = length(self.times)
        self.volData = np.array(volData) * 0.01
        self.totalVar = np.power(self.volData, 2.0) * self.times.reshape(-1, 1)
        self.logStrikes = np.log(strikes)
        self.fitter = [ssvi.Ssvi([-0.3, 0.01], phi.QuotientPhi([0.4, 0.4])) for i in range(self.slice_num)]
        self.vectorized_fitter = [np.vectorize(fitter[i]) for i in i in range(self.slice_num)]
        self.butterfly = ssvi.ssviQuotientConstraints
        if weight:
            self.weight = 5.0 * np.maximum(0.0, np.abs(np.log(weight_cut)) - np.abs(self.logStrikes))
        else:
            self.weight = np.repeat(1.0, length(self.logStrikes))
        self.params = [None for i in range(self.slice_num)]
        self.calendar_checker = np.linspace(-1.0, 1.0, 41)
        
    def cost_function(self, x, i):
        """
        cost_function(self, x)
        rho, theta, eta, gamma = x
        or
        rho, theta, lambda = x
        """
        self.fitter[i].reset(x)
        vf = np.vectorize(self.fitter[i])
        _value = np.sum( (self.weight*(self.totalVar[i] - vf(self.logStrikes)))**2.0 ) 
        return _value

    def calibrate_slice(self, i, init = np.array([-0.3, 0.01, 0.4, 0.4]),
                        method='SLSQP', maxiter = 10000, tol = 1.0e-16,
                        verbose = False):
        _init = init
        _cost_function = partial(self.cost_function, i)
        if i == self.slice_num - 1:
            res = minimize(
                self.cost_function, _init,
                constraints = self.butterfly,
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
        else:
            #current_fitter = copy.deepcopy(self.vectorized_fitter[i])
            #upper_fitter = copy.deepcopy(self.vectorized_fitter[i+1])
            _cost_function = partial(self.cost_function, i)
            def calendar(x):
                # rho, theta, eta, gamma
                x[0] -= constant.eps
                x[1] += constant.eps
                x[2] += constant.eps
                x[3] += constant.eps
                _fitter = copy.deepcopy(self.fitter[i])
                _fitter.reset(x)
                vf = np.vectorize(_fitter)
                ret = self.vectorized_fitter[i+1](self.calendar_checker) - constant.eps -vf(self.calendar_checker)
            calendar_cons = ({'type': 'ineq',
                     'fun': calendar}
                    )
            res = minimize(
                self.cost_function, _init,
                constraints = [self.butterfly, calendar_cons],
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
            
        self.fitter[i].reset(res.x)
        self.params[i] = res.x
        self.vectorized_fitter[i] = np.vectorize(self.fitter[i])
        
    def calibrate(self, init = np.array([-0.3, 0.01, 0.4, 0.4]),
                  method='SLSQP', maxiter = 10000, tol = 1.0e-16,
                  verbose = False):

        for i in range(self.slice_num-1, -1, -1):
            self.calibrate_slice(i, init, method, maxiter, tol, verbose)

    def fitted_vol(self, i, k):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt(self.vectorized_fitter[i](k) /self.t  )

    def fitted_slice(self, i):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt( self.vectorized_fitter[i](self.logStrikes) / self.t )

    def visualize_slice(self):
        st = np.exp(self.logStrikes)
        lst = self.logStrikes
        ax_size = int(self.slice_num/2.0)+1
        fig, axs = plt.subplots(ax_size, 2)
        count = 0
        
        for i in range(ax_size):
            axs[i, 0].plot(st, self.fitted_slice(count), 'r--')
            axs[i, 0].plot(st, self.volData[count]), 'b^')
            axs[i, 0].set_title("T ="+"{:10.2f}".format(self.times[count]))
            count += 1
            axs[i, 1].plot(st, self.fitted_slice(count), 'r--')
            axs[i, 1].plot(st, self.volData[count]), 'b^')
            axs[i, 1].set_title("T ="+"{:10.2f}".format(self.times[count]))
            count +=1
            
        plt.legend()
        plt.show()   
