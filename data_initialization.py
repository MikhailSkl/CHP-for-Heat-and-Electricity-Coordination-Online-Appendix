# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:40:59 2022

@author: mikska
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:58:21 2021

@author: mikska
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(886)

#Indexes
T=24;
CH=2;
G=2;
W=1;
S=50 #Number of electricity price forecast scenarious

time = ['t{0}'.format(t+1) for t in range(T)]
time_numbers=list(range(1,T+1))
gen=['G{0}'.format(g+1) for g in range(G)]
wind=['W{0}'.format(j+1) for j in range(W)]
CHP=['CHP{0}'.format(s+1) for s in range(CH)]
I=CHP+gen #Set of dispatchable units

#Self-scheduling CHPs
SSC=['CHP2']
#Non self-scheduling CHPs
NSS=['CHP1']
#Non-self scheduling units
N=['CHP1', 'G1', 'G2']

scenarious = ['s{0}'.format(w+1) for w in range(S)]

#Tecchnical charasteristics
heat_maxprod = {'CHP1': 300,'CHP2': 300} #CHP only

rho_elec = {'CHP1': 2.1,'CHP2': 2.1} # efficiency of the CHP for electricity production
rho_heat = {'CHP1': 0.25,'CHP2': 0.25} # efficiency of the CHP for heat production

r_chp = {'CHP1': 0.5,'CHP2': 0.5} # elec/heat ratio (flexible in the case of extraction units)

Fuel_max={'CHP1': 600,'CHP2': 600}
Fuel_min={'CHP1': 100,'CHP2': 100}

Ramp={}
Ramp['CHP1']=150
Ramp['CHP2']=150

#Generators
elec_maxprod = {} #Only for Generators
elec_maxprod['G1'] = 200
elec_maxprod['G2'] = 200

elec_minprod = {} #Only for Generators
elec_minprod['G1'] = 0
elec_minprod['G2'] = 100

Ramp['G1']=60
Ramp['G2']=100

#Cost
C_fuel={}
C_fuel['CHP1']=5 # elec = 15.75 heat 1.875
C_fuel['CHP2']=12.5 # elec = 21 heat 2.5

C_fuel['G1'] = 20.7 
C_fuel['G2'] = 6.7 

#Start-up cost
C_SU={}
C_SU['CHP1']=8000
C_SU['CHP2']=10000

C_SU['G1'] = 10000
C_SU['G2'] = 13000

#Initial status
P_ini={}
P_ini['CHP1'] = 100
P_ini['CHP2'] = 0 

P_ini['G1'] = 0
P_ini['G2'] = 0


U_ini={}
U_ini['CHP1'] = 1 
U_ini['CHP2'] = 0   

U_ini['G1'] = 0
U_ini['G2'] = 0



day = ['d{0:02d}'.format(t) for t in range(31)]

##Reading day-ahead data for DK1
# =============================================================================
data_madsen=pd.read_csv("veks.csv") 
# =============================================================================
# data_madsen.isnull().sum()
# data_madsen['HC.c']=data_madsen['HC.c'].dropna()
# data_madsen.isnull().sum()
# =============================================================================
heat_load_dec={}
for d in range(31):
    for t in range(T):
        heat_load_dec[day[d],time[t]]=data_madsen['HC.f'][24*d+t]*0.2777 #Convert from GJ/h to MW

heat_load_average = {t:(heat_load_dec[day[0],t]+heat_load_dec[day[2],t]+heat_load_dec[day[3],t])/3 for t in time}


heat_load_norm = {t:(heat_load_average[t]-min([heat_load_average[t] for t in time]))/(max([heat_load_average[t] for t in time])-min([heat_load_average[t] for t in time])) for t in time}

heat_load = {}

min1= 100 # 1500/2
max1= 500 # 2500/2
for t in time:
    heat_load[t]=heat_load_norm[t]*(max1-min1)+min1
    
    
# =============================================================================
# fig, ax = plt.subplots(1,1)  # Create a figure containing a single axes.
# keys=list(range(1,25,1))
values = heat_load.values()
# plt.bar(keys, values)
# ax.set_xlabel('Time Period')
# ax.set_title('Demand [MW]')
# =============================================================================

fig1, ax1 =plt.subplots(figsize = (4, 3))  # Create a figure containing a single axes.
plt.plot(time_numbers, list(values), label='Heat load')
#Electricity load
elec_load_IEEE = {'t1':820,'t2':820,'t3':815,'t4':815,'t5':810,'t6':850,'t7':1000,'t8':1150+100*1400/350 ,'t9':1250+100*1400/350 ,'t10':1250+100*1400/350 ,'t11':1100+100*1400/350,'t12':1000+100*1400/350,'t13':1000+50*1400/350,'t14':955+50*1400/350,'t15':950+50*1400/350,'t16':950+36*1400/350,'t17':900+25*1400/350,'t18':950,'t19':1010,'t20':1100,'t21':1125,'t22':1025,'t23':950,'t24':850}


elec_load_norm = {t:(elec_load_IEEE[t]-min([elec_load_IEEE[t] for t in time]))/(max([elec_load_IEEE[t] for t in time])-min([elec_load_IEEE[t] for t in time])) for t in time}

elec_load={}

min1_e= 350 # 1500/2
max1_e= 800 # 2500/2
for t in time:
    elec_load[t]=elec_load_norm[t]*(max1_e-min1_e)+min1_e



#fig2, ax2 = plt.subplots()  # Create a figure containing a single axes.
plt.plot(time_numbers, list(elec_load.values()), label='Electricity load')
plt.xlabel('Time Period')
plt.ylabel('Load [MW]')
#plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.xticks(np.arange(min(time_numbers)+2, max(time_numbers)+1, 3.0))
plt.savefig('loads.pdf', dpi=100, bbox_inches='tight')
plt.show() 

##Wind forecast DA
wind_DA = {}
             
wind_DA['W1','t1'] = 0.4994795107848
wind_DA['W1','t2'] = 0.494795107848
wind_DA['W1','t3'] = 0.494795107848
wind_DA['W1','t4'] = 0.505243011484
wind_DA['W1','t5'] = 0.53537368424
wind_DA['W1','t6'] = 0.555562455471
wind_DA['W1','t7'] = 0.628348636916
wind_DA['W1','t8'] = 0.6461954549
wind_DA['W1','t9'] = 0.622400860956
wind_DA['W1','t10'] = 0.580111023006
wind_DA['W1','t11'] = 0.714935503018
wind_DA['W1','t12'] = 0.754880140759
wind_DA['W1','t13'] = 0.416551027874
wind_DA['W1','t14'] = 0.418463919582
wind_DA['W1','t15'] = 0.39525842857
wind_DA['W1','t16'] = 0.523097379857
wind_DA['W1','t17'] = 0.476699300008
wind_DA['W1','t18'] = 0.626077589123
wind_DA['W1','t19'] = 0.684294396661
wind_DA['W1','t20'] = 0.0598119722706 
wind_DA['W1','t21'] = 0.0446453658917 
wind_DA['W1','t22'] = 0.485237701755
wind_DA['W1','t23'] = 0.49466503395
wind_DA['W1','t24'] = 0.4993958131342  


Wind_max={}
Wind_max['W1']=300
Wind_DA={}

for t in time:
    for j in wind:
        Wind_DA[j,t]= wind_DA[j,t]*Wind_max[j]


fig2, ax2 = plt.subplots(figsize = (4, 3))
plt.plot(time_numbers, list(Wind_DA.values()), label='Day-ahead wind forecast') 
plt.xlabel('Time Period')
plt.ylabel('Wind power [MW]')
#plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.xticks(np.arange(min(time_numbers)+2, max(time_numbers)+1, 3.0))
ax2.set_ylim(0, 270)
ax2.set_xlim(0, 25)
plt.savefig('wind.pdf', dpi=100, bbox_inches='tight')
plt.show() 
    
    
#Prices from integrated dispatch    
         
actual_price={'t1': 15.75,
 't2': 12.225,
 't3': 6.7,
 't4': 0.0,
 't5': 0.0,
 't6': 0.0,
 't7': 6.7,
 't8': 21.0,
 't9': 21.0,
 't10': 56.0,
 't11': 21.0,
 't12': 21.0,
 't13': 21.0,
 't14': 15.75,
 't15': 15.75,
 't16': 15.75,
 't17': 6.7,
 't18': 0.0,
 't19': 6.7,
 't20': 15.75,
 't21': 15.75,
 't22': 6.7,
 't23': 6.7,
 't24': 6.7}

forecast_price={'t1': 17.556284153005482,
 't2': 14.900506849315056,
 't3': 12.349010989010988,
 't4': 6.096082191780798,
 't5': 0.0,
 't6': 0.0,
 't7': 13.889589041095888,
 't8': 17.444054794520536,
 't9': 16.469095890410976,
 't10': 69.97536986301367,
 't11': 32.40969863013698,
 't12': 16.83630136986302,
 't13': 19.94843835616433,
 't14': 14.535753424657536,
 't15': 18.88983561643834,
 't16': 16.785150684931498,
 't17': 2.8461643835616544,
 't18': 0.0,
 't19': 16.35553424657535,
 't20': 11.975780821917851,
 't21': 9.924109589041112,
 't22': 6.660767123287688,
 't23': 3.301671232876708,
 't24': 0.0}

actual_price_h={'t1': 1.875,
 't2': 3.6375,
 't3': 6.4,
 't4': 13.0,
 't5': 13.0,
 't6': 13.0,
 't7': 9.65,
 't8': 2.5,
 't9': 2.5,
 't10': 6.6667,
 't11': 2.5,
 't12': 2.5,
 't13': 2.5,
 't14': 1.875,
 't15': 1.875,
 't16': 5.125,
 't17': 9.65,
 't18': 13.0,
 't19': 9.65,
 't20': 5.125,
 't21': 6.2083,
 't22': 9.65,
 't23': 9.65,
 't24': 6.4}
np.random.seed(886)
##Reading day-ahead data for DK1
# =============================================================================
day_ahead_prices=pd.read_excel("elspot_2017_eur.xlsx") 
# ts_delhi=day_ahead_prices[['DataStamp','DK1']]
# ts_delhi['DataStamp'] = pd.to_datetime(day_ahead_prices['DataStamp'])
# prs_day_avg = ts_delhi.resample('D', on='DataStamp').mean()
# prs_day_avg.plot(figsize = (15, 6))
# plt.show()
# =============================================================================
#Downloading data
day_ahead_DK1=day_ahead_prices[['DateStamp','DK1']]
day_ahead_DK1['DateStamp'] = pd.to_datetime(day_ahead_DK1['DateStamp'])
#day_ahead_DK1.loc(:,'DateStamp') = pd.to_datetime(day_ahead_DK1['DateStamp'])
#Add hours to the DateStamp
day_ahead_DK1['DateStamp'] += pd.to_timedelta(day_ahead_DK1.groupby('DateStamp').cumcount(), unit='H')

#day_ahead_DK1.plot(x='DateStamp', y='DK1', figsize = (14, 6), legend = True, color='g')

##Exclude December
price_not_dec=day_ahead_DK1.loc[day_ahead_DK1['DateStamp'].dt.month < 12]


T=24
S=50
error_t={}
mean_t={}
mean_dec={}
for t in range(0,T):
    #price_t=price_not_dec.loc[(price_not_dec['DateStamp'].dt.hour==t)]
    price_t=day_ahead_DK1.loc[(day_ahead_DK1['DateStamp'].dt.hour==t)]
    mean_t[t] = price_t['DK1'].mean()
    error_t[t] = price_t['DK1']-mean_t[t] 

# =============================================================================
# price_dec=day_ahead_DK1.loc[day_ahead_DK1['DateStamp'].dt.month==12 ]   
# for t in range(0,T):
#     price_t_dec=price_dec.loc[(day_ahead_DK1['DateStamp'].dt.hour==t)]
#     mean_dec[t] = price_t_dec['DK1'].mean()
#     #error_t[t] = price_t['DK1']-mean_t[t] 
# mean_dec=pd.DataFrame.from_dict(mean_dec, orient='index')
# mean_dec.plot()
# 
# 
# y = np.zeros(shape=(T, S))
# for w in range(0, S):
#     #print('w= ', w)
#     for t in range(0, T):
#         #print('t= ', t)
#         a=pd.DataFrame(error_t[t])
#         xi=a.sample(1)
#         y[t, w]=mean_dec.iloc[t]+xi['DK1'].values
# =============================================================================

#take just typical day in december and generate variation around it
        
#price_dec=day_ahead_DK1.loc[(day_ahead_DK1['DateStamp'].dt.month==12) & (day_ahead_DK1['DateStamp'].dt.day == 6 ), 'DK1' ]


#day15=day_ahead_DK1.loc[(day_ahead_DK1['DateStamp'].dt.month==12) & (day_ahead_DK1['DateStamp'].dt.day == 15 ), 'DK1' ]

#day15=pd.DataFrame(day15)
#day15.plot(label='Actual Price')


# =============================================================================
# y = np.zeros(shape=(T, S))
# for w in range(0, S):
#      #print('w= ', w)
#      for t in range(0, T):
#          #print('t= ', t)
#          a=pd.DataFrame(error_t[t])
#          xi=a.sample(1) #randomly sample from the past errors
#          y[t, w]=day15.iloc[t]+xi['DK1'].values
# =============================================================================
         
y = np.zeros(shape=(T, S))
for w in range(0, S):
     #print('w= ', w)
     for t in range(0, T):
         #print('t= ', t)
         a=pd.DataFrame(error_t[t])
         xi=a.sample(1) #randomly sample from the past errors
         y[t, w]=actual_price['t{0}'.format(t+1)]+xi['DK1'].values

#Replace negative prices with 0, we do not use negative prices in the model
y[y<0] = 0

fig1, ax1 = plt.subplots(figsize = (4, 3))
y_w={}    
for w in range(0,S):
    y_t=[]
    for t in range(0,T):
        #print('t= ', t, 'w= ', w, 'y=' , y['t{0}'.format(t), 'w{0}'.format(w)])
        y_t.append(y[t, w])            
    #print(y_t)
    if w==49:
        y_w['w{0}'.format(w)]=y_t
        plt.plot(y_w['w{0}'.format(w)], 'tab:gray', label='Price scenarios')
    else:
        y_w['w{0}'.format(w)]=y_t
        plt.plot(y_w['w{0}'.format(w)], 'tab:gray')
#day15=pd.DataFrame(day15)
plt.plot(pd.DataFrame(actual_price.values()), label='Electricity price')
plt.xlabel('Time Period')
plt.ylabel('Price [EUR\MWh]')
plt.legend(frameon=False)
plt.savefig('Scenarious.pdf', dpi=100, bbox_inches='tight')
plt.show() 


fig3, ax3 = plt.subplots(figsize = (4, 3)) 
plt.plot(time_numbers, pd.DataFrame(forecast_price.values()), label='Forecasted price')
plt.xlabel('Time Period')
plt.ylabel('Price [EUR\MWh]')
#plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.xticks(np.arange(min(time_numbers)+2, max(time_numbers)+1, 3.0))
ax3.set_ylim(-5, 85)
#ax2.set_xlim(0, 25)
#ax3.legend(loc='lower right')   
plt.savefig('EL_price_forecast.pdf', dpi=100, bbox_inches='tight')
plt.show() 


