import xlwings as xw
import utils
import fitting
import ssvi
import phi
from datetime import datetime, date
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
    fitter = [ssvi.Ssvi([-0.2, 0.02], phi.QuotientPhi([0.4, 0.4])) for i in range(len(dates))]
    surface = fitting.SurfaceFit(dt, dates, mult_money, vol, fitter,
                                 weight = weight,
                                 weight_cut = weight_cut,
                                 calendar_buffer = calendar_buffer)

    surface.calibrate(maxiter = max_iter, verbose = True)
    surface.visualize()
    
    msg = "데이터베이스에 입력 하시겠습니까?\n\n" 
    msg += "(원하지 않으실 경우 No를 누르시고,\n"
    msg += "weight_cut | calendar_buffer 값등을 조정해서 재 실행 해보시기 바랍니다. \n"
    msg += "default 값은 weight_cut=0.4 (=40%), calendar_buffer=0.0003 입니다. \n" 
    msg += "외 가격의 비중을 줄이려면 weight_cut값을 늘리고 (e.g., 0.7), \n" 
    msg += "calendar arbitrage 발생시 \n"
    msg += "calendar_buffer를 낮게 설정해 주시면 됩니다 (0.00015 미만은 권장되지 않음). \n" 
    msg += "[예시: fit_with_forward_moneyness(dt, dates, money, vol, TRUE, 0.7, 0.0002)] \n"
    msg += "[다섯번째 변수가, FLASE일 경우 가중치 부여 없음] \n" 
    msg += "8, 9 번째의 변수 값은 볼 스케일 값과 \n"
    msg += "(e.g., 데이터가 25.432등으로 들어올경우 0.01로 설정) \n" 
    msg += "최적화 시 max iteration값 입니다 \n"
    msg += "(e.g., 결과가 만족스럽지 않다면 20000으로 설정). \n\n" 
    msg += "주의! 엑셀 계산 옵션이 자동이라면 데이터 값 변경시 자동으로 실행 됩니다.\n"
    
    m_res = utils.Mbox("", msg, 4)
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

    
    

#import xlwings as xw;xw.serve()
    
