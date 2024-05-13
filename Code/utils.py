# -*- coding: utf-8 -*-
import numbers
import pandas as pd
import QuantLib as ql
import datetime as dt
import time
from datetime import date, datetime
import tkinter as tk
from tkinter import messagebox

def show_message_box(msg, title, width=300, height=100):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.geometry(f"{width}x{height}")
    messagebox.showinfo(title, msg)

def str2date(x):
    d = datetime.strptime(x, "%Y-%m-%d")
    return d.date()

def date2str(dt):
    ret = dt.strftime("%Y%m%d")
    return ret

def date2qldate(dt):
    ret = ql.Date(dt.day, dt.month, dt.year)
    return ret

def qldate2date(dt):
    ret = date(dt.year(), dt.month(), dt.dayOfMonth())
    return ret

def str2qldate(dt):
    """
    str2qldate(date)
    20200415 => ql.Date(15, 4, 2020)
    """
    ret = ql.Date(int(dt[-2:]), int(dt[-4:-2]),  int(dt[:4]))
    return ret

def qldate2str(dt):
    """
    by the format of 'yyyymmdd'
    """
    return date2str(qldate2date(dt))

def char_to_yearfrac(char):
    """
    char_to_yearfrac(char)
    'D' -> 1/365
    'W' -> 7*D
    'M' -> 1/12
    'Y' -> 1
    """
    assert char in ('D', 'W', 'M', 'Y', 'd', 'w', 'm', 'y'), (
            "error in char_to_yearfrac. The input is not either D/d/W/w/M/m/Y/y")
    if char in ('D' or 'd'):
        return 0.0027397260273972603
    elif char in ('W' or 'w'):
        return 0.019178082191780823
    elif char in ('M' or 'm'):
        return 0.08333333333333333
    else:
        return 1.

def string_to_yearfrac(string, reverse = False):
    """
    EX)
    string_to_yearfrac(string)=> '6M' -> 0.5
    string_to_yearfrac(string, reverse=True) => 'M6' -> 0.5
    """
    if reverse:
        assert string[0] in ('D', 'W', 'M', 'Y', 'd', 'w', 'm', 'y'), (
                "string_to_yearfrac. check the input is like 'W3', 'D3', etc.")
        return float(string[1:]) * char_to_yearfrac(string[0]) 
    else:
        assert string[-1] in ('D', 'W', 'M', 'Y', 'd', 'w', 'm', 'y'), (
                "string_to_yearfrac. check the input is like '3W', '3D', etc.")
        return float(string[:-1]) *char_to_yearfrac(string[-1])

def where_none(arr, column_wise = True):
    '''
    where_none(arr, column_wise = True)
    
    If column_wise = True, 
    this finds the first index where the array has None.
    If there is no None, this returns -1.
    
    If column_wise is not True, it finds where the None appear in the first row.
    '''
    ret=0
    if column_wise:
        column= [row[0] for row in arr]
        for c in column:
            if c == None:
                return ret
            else:
                ret = ret + 1
    else:
        for r in arr[0]:
            if r == None:
                return ret
            else:
                ret = ret + 1
    return -1


def datetime_yearfrac(st, ed):
    """
    yearfrac(st, ed)
    
    To find the yearfrac betwwen st and ed
    """
    assert isinstance(st, dt.datetime), "yearfrac, st is not datetime"
    assert isinstance(ed, dt.datetime), "yearfrac, ed is not datetime"

    d_st = date(st.year, st.month, st.day)
    d_ed = date(ed.year, ed.month, ed.day)
    delta = d_ed - d_st
    return delta.days/365.25

def head_date_to_string(arr, col_index=0):
    """
    head_date_to_string(arr)
    change the first column type from datetime to string
    e.g., 2020-01-02 => '20200102'
    """
    # this is just for checking type. No other purpose.
    for i in range(len(arr)):
        if isinstance( arr[i][col_index], dt.datetime) : 
            arr[i][col_index] = arr[i][col_index].strftime("%Y%m%d")
        if isinstance( arr[i][col_index],  str) or isinstance( arr[i][col_index],  numbers.Number):
            arr[i][col_index] = str( int( float(arr[i][col_index]) ) )


def not_sparse_row(arr, threshold=0.99):
    """
    return indices of rows if there are not too many nones. default = 80%
    """
    ret=[]
    def check_none(r):
        sz = len(r)
        cnt = r.count(None)
        return cnt/sz < threshold 
    for row in arr:
        ret.append(check_none(row))
    return ret
        

# Bloomberg error message 
BbgMessageInProgress = '#N/A Requesting Data...'

BbgMessageInvalid    = '#N/A Invalid Security'

BbgMsgNotApplicable  ='#N/A Field Not Applicable'

BbgMsgNA  ='#N/A'

def check_bloomberg_error(data, time_to_sleep):
    """
    check_bloomberg_error(data)

    To check whether the two dimensional array data 
    has bloomber error message.
    This returns True if it has an error msg, false otherwise.
    """

    for d in data:
        if BbgMessageInProgress in d \
           or BbgMessageInvalid in d \
           or BbgMsgNotApplicable in d \
           or BbgMsgNA in d:
            time.sleep( time_to_sleep )
            return True

    return False
        
def conversion(lst):
    res_dct = {item[0]: item[1] for item in lst}
    return res_dct

def json_from_db(query, conn, one=False):
    rows = conn.execute(query)
    cur = rows.cursor
    df = pd.read_sql_query(query, conn)
    return df.to_json()

def matrix2str(M):
    """
    matrix2str(m)
    [[1.0, 2], [None, '1']]
    => "[1.0, 2] \n [None, '1']"
    """
    return " ".join(str(x)+"\n" for x in M)





