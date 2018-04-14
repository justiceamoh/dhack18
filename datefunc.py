import pandas as pd

def funcdate(example):
    #### function 1&2:
    #####input pandas series and return date frame with year, month and day of week
    date=pd.to_datetime(example)
    dateframe=pd.DataFrame()
    dateframe['month'] = date.dt.month
    dateframe['day'] = date.dt.day
    dateframe['year'] =date.dt.year
    dateframe['dayofweek']=date.dt.dayofweek
    ####find a specific day (christmas) of the corresponding years and return time difference
    xmastime=pd.DataFrame()
    xmastime['year']=date.dt.year
    xmastime['month']=12
    xmastime['day']=25
    ####xdate is pandas series for christmas day
    xdate=pd.to_datetime(xmastime)
    dateframe['timediff']=(xdate-date).dt.days
    return dateframe