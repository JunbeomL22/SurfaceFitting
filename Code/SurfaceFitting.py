import xlwings as xw
import utils
import fitting
import ssvi
import phi
from datetime import datetime
import numpy as np
import unicodedata as ud

@xw.func
def fit_with_forward_moneyness(dt, dates, money, vol,
                               weight=True,
                               weight_cut = 0.4, calendar_buffer = 2.0e-4,
                               vol_scale =1.0, 
                               max_iter = 10000):
    if type(dates[0]) == str:
        dates = list(map(utils.str2date, dates))

    if type(dt) == datetime:
        dt = dt.date()
        
    if type(dt) == str:
        dt = utils.str2date(dt)

    vol = np.array(vol) * vol_scale
    mult_money = [money for i in range(len(dates))]
    #import pdb;pdb.set_trace()
    fitter = [ssvi.Ssvi([-0.3, 0.01], phi.QuotientPhi([0.4, 0.4])) for i in range(len(dates))]
    surface = fitting.SurfaceFit(dt, dates, mult_money, vol, fitter,
                                 weight = weight,
                                 weight_cut = weight_cut,
                                 calendar_buffer = calendar_buffer)

    surface.calibrate(maxiter = max_iter, verbose = True, method = 'SLSQP', tol = 5.0e-8)
    surface.visualize()
    
    new_msg = "If the result is not satisfactory, you may want to adjust the following inputs:\n"
    new_msg += " * weight_cut:\n"
    new_msg += "    This cuts off the deep otm data.\n"
    new_msg += "    For instance, if weight_cut = 0.7, the option data outside [0.7, 1.3] will be ignored.\n"
    new_msg += " * calendar_buffer:\n"
    new_msg += "    This is the buffer in applying the calendar arbitrage inequality constraint.\n"
    new_msg += "    If the calendar arbitrage is detected, you may want to lower this value.\n"
    new_msg += "    However, I recommend to set this value below 0.00015.\n"
    
    m_res = utils.show_message_box(new_msg, "", width = 500)

    if m_res == 6:
        rho = ud.lookup("GREEK SMALL LETTER RHO")
        theta = ud.lookup("GREEK SMALL LETTER THETA")
        eta = ud.lookup("GREEK SMALL LETTER ETA")
        gamma = ud.lookup("GREEK SMALL LETTER GAMMA")
        res = [[None, rho, theta, eta, gamma]]
        _params =surface.params
        
        for i, p in enumerate(_params):
            row = [dates[i]] + list(p)
            res = res + [row]
        return res
    else:
        return

