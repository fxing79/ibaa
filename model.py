import datetime
import requests
import xlrd
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import display

#data of daily price and the corresponding trading volume time series
dictaapl=
dictgs=
dictpfe=
dictnem=
dictsbux=
vdictaapl=
vdictgs=
vdictpfe=
vdictnem=
vdictsbux=

#This function performs the inverse optimization of views as in BL model, given the price in period (t) and in the future period (t+1).
#Optimized views can then be fed into supervised learning models.
#Inputs:
#--date--the index (t) of the optimized views 
#--delta--risk aversion coefficient as in CAPM model, recommended setting delta=2.5
#--tau--the confidence level of CAPM, recommended setting tau=0.05, if using 20 years data
#--pfolio--a list of tickers that are selected to construct the investigated portfolio
#--sentisource--the local excel file that contains raw sentiment data with regard to specific stocks
#--timespan--the time window used to calculate correlation between stocks in portfolio
#Outputs:
#--opti_w--the weights of the portfolio that generate maximum daily return 
#--Q--the intensity judge matrix as in BL model views PQOmega
def get_opti_views(date, delta, tau, pfolio, sentisource, timespan, case, method):
    weq = get_mkt_weq(date, pfolio, case)
    Omega = get_view_confience(date, pfolio, sentisource, method)
    [correts, annual_volatility] = get_corrcoef_vol(date, pfolio, timespan)
    V = np.multiply(np.outer(annual_volatility,annual_volatility)/(100**2), correts)
    pi = weq.dot(V * delta)
    ts = tau * V
    wopt = get_opti_w(date, pfolio)
    pmatrix = np.linalg.inv(np.linalg.inv(ts)+np.linalg.inv(np.diag(Omega)))
    confidence_adjusted_view_intensity =(np.array(wopt)*delta).dot(pmatrix+V).dot(np.linalg.inv(pmatrix))-np.linalg.inv(ts).dot(pi)
    return (wopt, confidence_adjusted_view_intensity.dot(np.diag(Omega)))
   
#Inputs:    
#--start_date-- the first day of the trading simulation
#--ttlasset-- the amount of total asset starting from the first day
#--pfolio--a list of tickers that are selected to construct the investigated portfolio
#--window-- the timespan for calculating stock correlations for simulation
#--autoviews-- computer generated views (the Q trianed) based on the NN output from raw sentiment, price, volume data etc.
def trading_simulation(start_date, ttlasset, pfolio, window, autoviews, sentisource, method):
    line=[ttlasset]
    i=0
    for q in open(autoviews):
        pratio=get_price_ratio(start_date+i,pf) 
        weight=blsenti(start_date+i-1, pf, q, sentisource, method)
        nweight=normalize(weight)
        asset_dis=nweight*line[-1]
        line.append(asset_dis.dot(pratio))
        i=i+1
    return line

def trading_simulation_vw(start_date, ttlasset, pfolio, window):
    line=[ttlasset]
    for i in range (window):
        pratio=get_price_ratio(start_date+i,pf) 
        asset_dis=get_mkt_weq(start_date+i-1, pfolio, 'specify_date')*line[-1]
        line.append(asset_dis.dot(pratio))
    return line

def blsenti(date, pfolio, Q, sentisource, method): 
    weq=get_mkt_weq(date, pfolio, 'specify_date')
    Omega=get_view_confience(date, pfolio, sentisource, method)
    Qls=[float(x) for x in Q.split(",")]
    [correts, annual_volatility]=get_corrcoef_vol(date, pfolio, timespan)
    V = np.multiply(np.outer(annual_volatility,annual_volatility)/10000, correts)
    pi = weq.dot(V * delta) #refPi
    ts = tau * V
    pmatrix= np.linalg.inv(np.linalg.inv(ts)+np.linalg.inv(np.diag(Omega)))
    sigmabar= V+pmatrix
    mubar=pmatrix.dot(np.linalg.inv(ts).dot(pi)+np.linalg.inv(np.diag(Omega)).dot(Qls))
    w=np.linalg.inv(sigmabar).dot(mubar)/delta
    return w

def getsentiment(date, pfolio, sentisource):    
    senti_repo=xlrd.open_workbook(sentisource)
    psenti=[]
    for asset in pfolio:
        table=senti_repo.sheet_by_name(asset)
        try:
            sentiment=table.row_values(table.col_values(1).index(date-693594))
            psenti.extend(sentiment[2:6])
        except ValueError:
            psenti.extend([0.0, 0.0, 0.0, 0.0])
    return (psenti)
                      
def get_view_confience(date, pfolio, sentisource, method):
    vconf=[]
    if (method=='omega0'):
        [correts, annual_volatility]=get_corrcoef_vol(date-1, pfolio, timespan)
        V = np.multiply(np.outer(annual_volatility,annual_volatility), correts)
        vconf=np.diag(tau*V)
    else:
        print ('Warning: this method is not implemented')
    return (vconf)
    
def Q_lstm(starting_date, len_history, pfolio, sentisource, timespan, add_sentiment):
    dataset = []
    Q=[]
    if (add_sentiment==0):
        trn_len=50
    if (add_sentiment==1):
        trn_len=70
    for i in range(len_history):
        with open('qlstm.txt','a') as f:
            print ('Approximating Q of day '+str(i+1)+' in '+ str(len_history)+' days...')
            dataset.append(get_training_record(starting_date+i, pfolio, sentisource, timespan, add_sentiment))
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(np.array(dataset)[:,0:trn_len])
            trainY=np.array(dataset)[:,trn_len:trn_len+5]
            trainX= np.reshape(data,(data.shape[0],data.shape[1],1))
            #training model
            batch_size = 1
            time_step=70
            model = Sequential()
            model.add(LSTM(3, batch_input_shape=(batch_size, time_step, 1), stateful=True, return_sequences=True))
            model.add(LSTM(3, batch_input_shape=(batch_size, time_step, 1), stateful=True))
            model.add(Dense(5))
            model.compile(loss='mean_squared_error', optimizer='rmsprop')
            model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=0, shuffle=False)
            #test data
            testd= np.array(get_training_record(starting_date+i+1, pfolio, sentisource, timespan, add_sentiment)[0:trn_len])
            testd=np.reshape(testd, (1,trn_len,1))
            f.writelines(str(model.predict(testd, batch_size=1).tolist())[2:-2]+'\n')
            
#This function computes the correlation and covariance matrices from (date-timespan) to (date)
#we recommend timespan should not be less than 30 days for estimation accuracy
def get_corrcoef_vol(date, pfolio, timespan):
    dailyreturns= []
    dailyprices= []
    for i in range(timespan):
        dailyreturns.append(np.array([getreturn(dictaapl, date-i),getreturn(dictgs, date-i),getreturn(dictpfe, date-i),getreturn(dictnem, date-i),getreturn(dictsbux, date-i)]))
        dailyprices.append(np.array([getprice(dictaapl, date-i),getprice(dictgs, date-i),getprice(dictpfe, date-i),getprice(dictnem, date-i),getprice(dictsbux, date-i)]))
    return (np.corrcoef(np.array(dailyreturns).T), np.std(dailyprices, axis=0)*np.sqrt(365/float(timespan)))  

def get_mkt_weq(date, pfolio, case):
    caps=[]
    if  case=='specify_date':  #data from excel file, this is always recommended
        cap_repo=xlrd.open_workbook('marketcap.xlsx')
        pcap=[]
        for asset in pfolio:
            table=cap_repo.sheet_by_name(asset)
            dm=date
            while True:
                try:
                    pcap.append(table.row_values(table.col_values(0).index(dm-693594))[1])
                    break
                except ValueError:
                    dm=dm-1
        weq=np.array(pcap)/sum(pcap)
    else :   
        weq=[]
        print ('Warning: method of getting equilibrium market weights not implemented!')
    return weq

def getprice(dict, d):
    try:
        price=dict[datetime.date.fromordinal(d).isoformat()]
        #recover price from split 
        if (dict==dictaapl and d<735393): 
            return  price/7
        elif (dict==dictsbux and d<735697):
            return price/2
        else:
            return price
    except KeyError: 
        price=getprice(dict, d-1)
        return price
    
def gettvolume(dict, d):
    try:
        tvolume=dict[datetime.date.fromordinal(d).isoformat()]
    except KeyError: 
        tvolume=gettvolume(dict, d-1)
    return tvolume

def get_price_ratio(date, pfolio):
    pricedate=np.array([getprice(dictaapl, date),getprice(dictgs, date),getprice(dictpfe, date),getprice(dictnem, date),getprice(dictsbux, date)])
    pricetmr=np.array([getprice(dictaapl, date+1),getprice(dictgs, date+1),getprice(dictpfe, date+1),getprice(dictnem, date+1),getprice(dictsbux, date+1)])
    return (pricetmr/pricedate)

def getreturn(dict, d):
    return (getprice(dict, d)-getprice(dict, d-1))

def parse_ymd(s):
    year_s, mon_s, day_s = s.split('-')
    return datetime.date(int(year_s), int(mon_s), int(day_s))

def get_opti_w(date, pfolio):
    price_ratio=list(get_price_ratio(date, pfolio))
    opti_w=[0]*len(pfolio)
    opti_w[price_ratio.index(max(price_ratio))]=1
    return (opti_w)  

def normalize(weight):
    if min(weight)<0:
        wpie=np.array(weight)-min(weight)
    else:
        wpie=np.array(weight)
    wpie=wpie/(sum(wpie))
    return (wpie)

def get_training_record(date, pfolio, sentisource, timespan, add_sentiment):
    [opt_weights, q_gt]=get_opti_views(date, delta, tau, pfolio, sentisource, timespan,'specify_date', 'omega0')
    if (add_sentiment==0):
        return (inputformat(date)+list(q_gt))
    elif(add_sentiment==1):
        return (inputformat(date)+getsentiment(date, pf, source)+list(q_gt))
    else:
        print ('Unexpected input variable')
        return 0
        
#define parameters
delta=0.25
tau=0.05
timespan=90
pf=['AAPL','GS','PFE','NEM','SBUX']
source='PsychSigSentiment.xlsx'

#Using Example 1: Inspecting the optimal views
d=736481 #the calendar date of '2017-06-01'
[opt_weights, Q]=get_opti_views(d, delta, tau, pf, source, timespan,'specify_date', 'omega0')
confidence=get_view_confience(d, pf, source, 'omega0')
print ('the optimal weights for '+datetime.date.fromordinal(d).isoformat()+' are: '+str(opt_weights))
print ('-----Optimal market views-----')
for i in range(len(Q)):
    if (np.abs(Q[i])>0.01):
        print ('I have '+'%.2f%%' % (confidence[i])+' confidence that '+pf[i]+' will outperform market by '+'%.2f%%' % (Q[i]*100))
print ('------------------------------')
print ('*Note that the "optimal views" can never be reached, because knowing the asset prices of tomorrow is impossible')

#Using Example 2: The trading simulation of market following strategy
#This example may take 5-10 minutes to finish
VW=[]
VW=trading_simulation_vw(733685, 10000, pf, 20)
print ('-----The trading simulation of market following strategy (VW)-----')
plt.plot(VW, label='VW')
plt.legend(loc='best')
plt.show()

#Using Example 3: The trading simulation of a sentic BL model, using <model=LSTM, timespan=90>
timespan=90
#Estimating Q may take extremely long time (even weeks) depending on epoch setting
Q_lstm(733685, 20, pf,source, timespan,1)
line=trading_simulation(733685 , 10000, pf, 20,'qlstm.txt', source, 'omega0')
print ('-----The trading simulation of LSTM(BL+s): first 20 days-----')
plt.plot(VW[0:20], color='g', label='VW',lw=1)
plt.plot(line, color='r', label='LSTM(BL+S)',ls='-.',lw=1)
plt.legend(loc='best')
plt.show()
