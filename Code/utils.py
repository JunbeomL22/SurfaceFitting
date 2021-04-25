# -*- coding: utf-8 -*-
import numbers
import sqlalchemy
import pandas as pd
import numpy as np
import xlwings as xw
import QuantLib as ql
import datetime as dt
import ctypes
import pdb
from sqlalchemy import MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from math import isnan
import time
from datetime import date, datetime

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

def db_engine(database, schema = "OTCUSER", password = "otcuser"):
    '''
    The input is the schema name, i.e., either one of otcora or otcora_114.
    Then it retuns the engine object in sqlalchemy.
    ex) 
    >>> engine = db_engine("otcora")
    >>> sql_query = pd.read_sql_query("select * from otc_ir_daily_val fetch first 10 rows only", conn)
    >>> sql_query
        IR_CODE BASE_DATE     DAY   YEAR     VALUE
        0       5190110  20120813   365.0   1.00  0.028800
        1       5190110  20120813   730.0   2.00  0.029200
        2       5190110  20120813  1095.0   3.00  0.030500
        3       5190110  20120813  1825.0   5.00  0.031600
        4       5190110  20120813  3650.0  10.00  0.032700
        5       5190110S  20120813    91.0   0.25  2.829965
        6       5190110S  20120813   274.0   0.75  2.859872
        7       5190110S  20120813   548.0   1.50  2.889973
        8       5190110S  20120813   913.0   2.50  2.971423
        9       5190110S  20120813  1460.0   4.00  3.094200
    '''
    DATABASE = database
    SCHEMA   = schema
    PASSWORD = password
    connstr  = "oracle://{}:{}@{}".format(SCHEMA, PASSWORD, DATABASE)
    engine = sqlalchemy.create_engine(connstr, max_identifier_length=128)
    return engine

def get_holidays(country_code, days = 365):
    """
    get_holidays(country_code, days = 365)
    #
    country_code = 'KR', 'US', 'HK', 'UK', 'EU', etc.
    To get list of holidays since 'days' (input variable) ago (default is 365).
    """
    engine = db_engine('otcora')
    conn = engine.connect()
    ql_2year_ago = date2qldate(date.today()) - ql.Period(days, ql.Days)
    st_dt = qldate2str(ql_2year_ago)
    country_code = '\'' + country_code + '\''
    condition  = " where market = " + country_code + " and  "
    condition = condition + "is_weekend is null and non_trading=1 and base_date > " +st_dt

    holidays = conn.execute("select * from otc_market_calendar" + condition)
    ret = []
    for d in holidays: ret.append(d[0])
    engine.dispose()
    
    return ret

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

def Mbox(title, text, style):
    '''
    Mbox(title, text, style)
    ##  Styles:
    ##  0 : OK
    ##  1 : OK | Cancel
    ##  2 : Abort | Retry | Ignore
    ##  3 : Yes | No | Cancel
    ##  4 : Yes | No
    ##  5 : Retry | No 
    ##  6 : Cancel | Try Again | Continue

    '''
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

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
    

def raise_bloomberg_error(data, data_name= " "):
    """
    check_bloomberg_error(data, data_name)

    This raises exception if one of the data is either
    '#N/A Invalid Security' or '#N/A Invalid Security'

    data is a 2-dim array, and data_name is a string
    """
    for d in data:
        if BbgMessageInProgress in d:
            Mbox("", BbgMessageInProgress +" appears while pulling " + data_name, 0)
            raise Exception(BbgMessageInProgress +" appears while pulling " + data_name)
        elif BbgMessageInvalid in d:
            Mbox("", BbgMessageInvalid +" appears while pulling " + data_name, 0)
            raise Exception(BbgMessageInvalid +" appears while pulling " + data_name)
        elif BbgMsgNotApplicable in d:
            Mbox("", BbgMsgNotApplicable +" appears while pulling " + data_name, 0)
            raise Exception(BbgMsgNotApplicable +" appears while pulling " + data_name)
        elif BbgMsgNA in d:
            Mbox("", BbgMsgNotApplicable +" appears while pulling " + data_name, 0)
            raise Exception(BbgMsgNotApplicable +" appears while pulling " + data_name)
    pass

def upsert(database, table_name, df, engine, session):
    """
    upsert(database, table_name, df, schema='OTCUSER', password='otcuser')
    all inputs are string except df
    df should be a pd.DataFrame object
    """
    assert isinstance(df, pd.DataFrame), "upsert error, df is not dataframe"

    
    # make the column lower case
    df.columns = map(str.lower, df.columns)
    
    dict_val = df.to_dict(orient='records')

    for i, d in enumerate(dict_val):
        dict_val[i] = {k: d[k] for k in d if pd.notnull(d[k])}
    
    meta = sqlalchemy.MetaData()

    meta.reflect(engine, autoload=True)

    table = meta.tables[table_name]
    
    Base = declarative_base()
    class Model(Base): __table__ = table

    my_model =[]
    for d in dict_val:
        my_model.append(Model(**d))
    #if table_name == 'ficc_cap_atm_vol':
    #    pdb.set_trace()    
    for m in my_model:
        session.merge(m)

    session.commit()
    
class JbException(Exception):
    def __init__(self, message, number=0):
        self.message = message

    def __str__(self):
        return self.message + "\n" + str(self.number) + " occured."
    
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





