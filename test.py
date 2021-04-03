import fitting
import ssvi
from datetime import date, timedelta

calc_date = date(2021, 3, 25)
d1  = date(2021, 5, 21)
d2  = date(2021, 6, 18)
t1 = (d1 - calc_date).days / 365.0
t2 = (d2 - calc_date).days / 365.0
strikes = [0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2]
#log_strikes = log.(strikes)
vol1 = [34.28, 25.0, 20.75, 18.47, 16.15, 14.27, 13.2, 13.36, 18.49]
vol2 = [32.55, 24.72, 21.02, 19.02, 17.02, 15.30, 14.12, 13.58, 16.53]

slice1 = fitting.SliceFit(t2, strikes, vol2)#, constraints={})
slice1.calibrate(init= [-0.3, 0.01, 0.4, 0.4], method = 'SLSQP', verbose = False, tol = 1.0e-16)
#print(slice1.params)
slice1.visualize()
