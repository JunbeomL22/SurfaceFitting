import fitting
import ssvi
import phi
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import utils

calc_date = date(2014, 1, 2)
s = ["02/01/2014", "03/03/2014", "04/02/2014", "07/01/2014", "12/28/2014", "06/26/2015", "12/23/2015"]

dates = list(map(utils.str_to_date, s))

strikes = [0.80, 0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15, 1.20]

vol=[[37.079, 31.291, 25.594, 20.273, 18.307, 17.313, 17.153, 16.86, 17.334, 20.602, 22.576],
     [28.645, 24.926, 21.473, 18.627, 17.371, 16.655, 16.356, 15.627, 15.035, 15.338, 16.453],
     [26.588, 23.477, 20.698, 18.316, 17.231, 16.664, 16.478, 15.776, 14.917, 14.765, 15.292],
     [22.749, 20.759, 19.041, 17.375, 16.55, 16.422, 16.935, 16.335, 15.445, 14.904, 14.632],
     [21.865, 20.534, 19.305, 18.126, 17.61, 17.287, 18.437, 17.866, 17.047, 16.421, 15.977],
     [20.877, 19.755, 18.662, 17.584, 17.159, 17.053, 19.239, 18.69, 17.945, 17.339, 16.843],
     [20.517, 19.539, 18.595, 17.645, 17.281, 17.296, 19.942, 19.393, 18.689, 18.115, 17.643]]


fitter = [ssvi.Ssvi([-0.3, 0.01], phi.QuotientPhi([0.4, 0.4])) for i in range(len(dates))]
surface = fitting.SurfaceFit(calc_date, dates, strikes, vol, fitter,
                             weight_cut = 0.7,
                             calendar_buffer = 0.001) 
surface.calibrate(maxiter = 20000, verbose = False)
surface.visualize()
