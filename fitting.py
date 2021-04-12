import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ssvi
import phi
from functools import partial
import copy
from datetime import date
import constant
from mpl_toolkits import mplot3d

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
                 fitter,
                 weight = True,
                 weight_cut = 0.6,
                 calendar_buffer=0.001):
        diff = np.array(dates) - calc_date
        self.times = np.array([x.days/365.0 for x in diff])
        self.slice_num = len(self.times)
        self.volData = np.array(volData) * 0.01
        self.totalVar = np.power(self.volData, 2.0) * self.times.reshape(-1, 1)
        self.logStrikes = np.log(strikes)
        self.fitter = fitter
        self.vectorized_fitter = [np.vectorize(fitter[i]) for i in range(self.slice_num)]
        self.vectorized_g = [np.vectorize(fitter[i]) for i in range(self.slice_num)]
        self.butterfly = ssvi.ssviQuotientConstraints
        self.calendar_buffer = calendar_buffer
        if weight:
            self.weight = 5.0 * np.maximum(0.0, np.abs(np.log(weight_cut)) - np.abs(self.logStrikes))
        else:
            self.weight = np.repeat(1.0, len(self.logStrikes))
        self.params = [None for i in range(self.slice_num)]
        self.calendar_checker = np.linspace(-1.5, 1.5, 15)
        self.calendar_ox =  ['O' for i in range(self.slice_num)]
        self.calendar_ox[-1] = 'Nil'
        self.butterfly_ox = ['O' for i in range(self.slice_num)]
        
    def cost_function(self, i, x):
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
                _cost_function, _init,
                constraints = self.butterfly,
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
        else:
            _cost_function = partial(self.cost_function, i)
            cons = copy.deepcopy(self.butterfly)
            for z in self.calendar_checker:
                def calendar(k, x):
                    _fitter = copy.deepcopy(self.fitter[i])
                    _fitter.reset(x)
                    ret = self.fitter[i+1](k) - self.calendar_buffer - _fitter(k)
                    return ret
                
                cons.append({'type': 'ineq', 'fun': partial(calendar, z)})
            
            res = minimize(
                _cost_function, _init,
                constraints = cons,
                method = method,
                options = {'disp': verbose, 'maxiter': maxiter},
                tol = tol
            )
            
        self.fitter[i].reset(res.x)
        self.params[i] = res.x
        self.vectorized_fitter[i] = np.vectorize(self.fitter[i])
        self.vectorized_g[i] = np.vectorize(partial(self.g, i))
        
    def calibrate(self, init = np.array([-0.3, 0.01, 0.4, 0.4]),
                  method='SLSQP', maxiter = 10000, tol = 1.0e-16,
                  verbose = False):

        for i in range(self.slice_num-1, -1, -1):
            self.calibrate_slice(i, init, method, maxiter, tol, verbose)

        self.check_butterfly()
        self.check_calendar()
        
    def fitted_vol(self, i, k):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt(self.vectorized_fitter[i](k) /self.times[i]  )

    def fitted_slice(self, i):
        """
        fitted_vol(self, k)
        volatility point at k
        Note: k is log strike
        """
        return np.sqrt( self.vectorized_fitter[i](self.logStrikes) / self.times[i] )
    
    def g(self, i, k):
        """
        density(self, i, k):
        density of i-th slice
        """
        w = self.fitter[i]
        wk = w(k)
        eps = 1.0e-4
        w_first = ( w(k+eps) - w(k-eps) ) / (2.0*eps)
        w_second= ( w(k+eps) + w(k-eps) - 2.0*w(k) ) / (eps**2.0)
        g = (1.0 - k*w_first /(2.0*wk))**2.0 - 0.25*w_first**2.0*(1.0/wk + 0.25) + 0.25*w_second
        return g

    def visualize(self):
        st = np.exp(self.logStrikes)
        lst = self.logStrikes
        testing = [-1.0 + 0.01*i for i in range(201)]
        ax_size = max(int(np.floor( max(self.slice_num-1, 0) /4)  ) + 1, 2)
        T, St   = np.meshgrid(self.times, np.exp(self.logStrikes))
        Index, Lst= np.meshgrid(range(self.slice_num), self.logStrikes)
        fv = np.vectorize(self.fitted_vol)
        
        # Total Variance
        fig = plt.figure()
        fig.set_size_inches(8.0, 6.0)
        for i in range(self.slice_num-1, -1, -1):
            plt.plot(testing, self.vectorized_fitter[i](testing), '-',
                     linewidth = 0.8, label=f"T="+"{:2.2f}".format(self.times[i]))

        plt.legend(shadow = True, fancybox=True, loc = "upper center")
        plt.title("Total Variance")
        plt.grid(linestyle="--", linewidth = 0.1, color='black')
        fig.tight_layout()
        
        # surface 3D
        fig = plt.figure()
        fig.set_size_inches(7.0, 5.0)
        ax  = plt.axes(projection='3d')
        ax.plot_surface(St, T, fv(Index, Lst), rstride=1, cstride=1,
                       cmap='viridis', edgecolor='none')
        ax.set_xlabel('K')
        ax.set_ylabel('T')
        ax.set_zlabel('Vol')
        fig.tight_layout()
        
        """
        fig, axs = plt.subplots(ax_size, 4)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        fig.set_size_inches(18.0, 9.0)
        
        count= 0
        row  = 0
        col  = 0
        while count < self.slice_num:
            col = count%4
            row = int(np.floor((count / 4)))
            axs[row, col].plot(testing, self.vectorized_g[count](testing), 'r--', linewidth = 2, label='g(k)')
            axs[row, col].plot(testing, np.zeros(len(testing)), 'b-', linewidth=1)
            title = f"T="+"{:2.2f}".format(self.times[count])
            title +=f" (Butterfly: {self.butterfly_ox[count]}, Calendar: {self.calendar_ox[count]})"
            axs[row, col].set_title(title)
            axs[row, col].legend(shadow = True, fancybox=True, loc = "upper right")
            count += 1
        """

        fig, axs = plt.subplots(ax_size, 4)
        fig.subplots_adjust(wspace=0.2)
        fig.set_size_inches(16.0, 9.0, forward=True)
        count= 0
        row  = 0
        col  = 0
        while count < self.slice_num:
            col = count%4
            row = int(np.floor((count / 4)))
            axs[row, col].plot(st, self.fitted_slice(count), 'r--', linewidth = 2, label='(s)svi')
            axs[row, col].plot(st, self.volData[count], 'b^', markersize=4, label='data')
            title = f"T="+"{:2.2f}".format(self.times[count])
            title +=f" (Butterfly: {self.butterfly_ox[count]}, Calendar: {self.calendar_ox[count]})"
            axs[row, col].set_title(title)
            axs[row, col].legend(shadow = True, fancybox=True, loc = "upper right")
            axs[row, col].grid(linestyle="--", linewidth = 0.2, color='black')
            count += 1
        fig.tight_layout()

        plt.show()

    def check_calendar(self):
        if self.slice_num == 1:
            return 
        
        testing = [-2.0 + 0.001*i for i in range(4001)]
        for i in range(self.slice_num-1):
            if any(self.vectorized_fitter[i](testing) > self.vectorized_fitter[i+1](testing)):
                self.calendar_ox[i] = 'X'

    def check_butterfly(self):
        testing = [-2.0 + 0.005*i for i in range(4001)]
        for i in range(self.slice_num):
            if any(self.vectorized_g[i](testing) < 0.0):
                self.butterfly_ox[i] = 'X'

    def check_arbitrage(self):
        print("surface butterfly: ", self.check_butterfly())
        print("surface calendars: " , self.check_calendar())
