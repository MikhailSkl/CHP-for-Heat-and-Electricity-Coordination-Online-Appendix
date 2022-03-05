# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:40:15 2022

@author: mikska
"""
import data_initialization as indata
import math
from int_func import int_dispatch_function
from int_func_UC import int_dispatch_function_UC
from seq_func import seq_dispatch_func
from seq_func_UC import seq_dispatch_func_UC
from SS_func import SS_dispatch_func
from SS_func_UC import SS_dispatch_func_UC
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

##Day-ahead electricity price forecast

S=50 #Number of electricity price forecast scenarious
T=24

scenarious = ['s{0}'.format(s+1) for s in range(S)]
time = ['t{0}'.format(t+1) for t in range(T)]

elec_price_s=indata.y
elec_price_actual=indata.actual_price
Price_DA_E_F={}
Price_DA_dict={}
for w, w1 in zip(scenarious, range(0,S)):
    Price_DA_dict[w]={}
    for t, t1 in zip(time, range(0,T)):
        #print('t= ', t1, 'w=', w1)
        Price_DA_E_F[w,t]= elec_price_s[t1,w1]
        Price_DA_dict[w][t] = elec_price_s[t1,w1]
 
# =============================================================================
# =============================================================================
# # #from Int_dispatch import total_cost1, lambda_DA_E, lambda_DA_H
# # from int_dispatch_func import int_dispatch_function
(Wind_CT_int, total_cost_int, lambda_DA_E, lambda_DA_H) = int_dispatch_function(indata)
# # 
# # from seq_dispatch import seq_dispatch_func
(Wind_CT_seq, total_cost_seq, lambda_DA_E_seq, lambda_DA_H_seq) = seq_dispatch_func(indata, Price_DA_dict['s36'])
# # 
# # from SS_function import SS_dispatch_func
(Wind_CT_SS, total_cost_SS, lambda_DA_E_SS, lambda_DA_H_SS, u_DA_SS) = SS_dispatch_func(indata, Price_DA_dict['s36'], 's36')
# # 
# #Cost vs Scenario
# fig3, ax3 = plt.subplots()
# #plt.scatter(list(range(1,S+1)), cost_SS_s.values(), label="SS")
# #plt.scatter(list(range(1,S+1)), cost_seq_s.values(), label="Seq")
# plt.plot(list(lambda_DA_E.values()), label='Int') 
# plt.plot(list(lambda_DA_E_seq.values()), label='Seq') 
# plt.plot(list(lambda_DA_E_SS.values()), label='SS') 
# ax3.set_title('Electricity price')
# ax3.set_xlabel('Time Period')
# ax3.set_ylabel('Price EUR/MWh') 
# ax3.legend(loc='upper left')   
# plt.show()
# fig4, ax4 = plt.subplots()
# #plt.scatter(list(range(1,S+1)), cost_SS_s.values(), label="SS")
# #plt.scatter(list(range(1,S+1)), cost_seq_s.values(), label="Seq")
# plt.plot(list(lambda_DA_H.values()), label='Int') 
# plt.plot(list(lambda_DA_H_seq.values()), label='Seq') 
# plt.plot(list(lambda_DA_H_SS.values()), label='SS') 
# ax4.set_title('Heat price')
# ax4.set_xlabel('Time Period')
# ax4.set_ylabel('Price EUR/MWh') 
# ax4.legend(loc='upper left')   
# plt.show()
# =============================================================================
# Unit Commitment
# # from int_dispatch_func import int_dispatch_function
#(Wind_CT_int_UC, total_cost_int_UC) = int_dispatch_function_UC(indata) 
# from seq_dispatch import seq_dispatch_func
#(Wind_CT_seq_UC, total_cost_seq_UC) = seq_dispatch_func_UC(indata, Price_DA_dict['s36'])
#Ceiling U_DA_SS
# =============================================================================
# u_SS_round={}
# for t in indata.time:
#     for i in indata.I:
#         if u_DA_SS[i,t] > 0.01:
#             u_SS_round[i,t]=1
#         else:
#             u_SS_round[i,t]=0
# =============================================================================
# from SS_function import SS_dispatch_func
#(Wind_CT_SS_UC, total_cost_SS_UC) = SS_dispatch_func_UC(indata, Price_DA_dict['s36'], 's36', u_SS_round)
# # 


wind_CT_seq_s={}
cost_seq_s={}
price_el_seq_s={}
price_h_seq_s={}

wind_CT_SS_s={}
cost_SS_s={}
price_el_SS_s={}
price_h_SS_s={}
u_DA_SS={}

(wind_CT_int, total_cost_int, lambda_DA_E, lambda_DA_H) = int_dispatch_function(indata)
#s='s29'
#(cost_SS_s[s], price_el_SS_s[s], price_h_SS_s[s]) = SS_dispatch_func(indata, Price_DA_dict[s], s)
#============================================================================

# =============================================================================
scenarious_part1 = ['s{0}'.format(s+1) for s in range(0,S)]
# =============================================================================
# for s in scenarious_part1:
#       print(('-'+s)*30)   
#       print(s*30)   
#       (wind_CT_SS_s[s], cost_SS_s[s], price_el_SS_s[s], price_h_SS_s[s], u_DA_SS[s]) = SS_dispatch_func(indata, Price_DA_dict[s], s)  
# =============================================================================


# =============================================================================
# for s in scenarious_part1:    
#      (wind_CT_seq_s[s], cost_seq_s[s], price_el_seq_s[s], price_h_seq_s[s]) = seq_dispatch_func(indata, Price_DA_dict[s])     
#      print(s)
# 
# =============================================================================
RMSE_seq={}
RMSE_ss={}
bias_seq={}
bias_ss={}
for s in scenarious_part1:
    E_ss={}
    E_seq={}
    E_ss2={}
    E_seq2={}
    for t in time:
        E_ss[t]=(price_el_SS_s[s][t]-Price_DA_dict[s][t])
        E_seq[t]=(price_el_seq_s[s][t]-Price_DA_dict[s][t])
        E_ss2[t]=(price_el_SS_s[s][t]-Price_DA_dict[s][t])**2
        E_seq2[t]=(price_el_seq_s[s][t]-Price_DA_dict[s][t])**2
    RMSE_ss[s]=math.sqrt(1/T*sum(E_ss2.values()))
    RMSE_seq[s]=math.sqrt(1/T*sum(E_seq2.values()))
    bias_ss[s]=1/T*sum(E_ss.values())
    bias_seq[s]=1/T*sum(E_seq.values())

fig, ax = plt.subplots(figsize = (4, 3))
ax.scatter(RMSE_ss.values(), cost_SS_s.values(), facecolors='none', edgecolors='r',label="Proposed")
ax.scatter(RMSE_seq.values(), cost_seq_s.values(), label="Sequential")
ax.plot([1,26], [total_cost_int, total_cost_int], color='k', label='Integrated' )
#ax.set_title('Cost vs RMSE')
ax.set_xlabel('RMSE [EUR/MWh]') 
ax.set_xlim(5, 25)
ax.set_ylim(132000, 149000)
scale_y = 1e3
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
ax.yaxis.set_major_formatter(ticks_y)
ax.set_ylabel('Cost [kEUR]')
ax.legend(loc='upper left', frameon=False)   
plt.savefig('Cost_RMSE.pdf', dpi=100, bbox_inches='tight')
plt.show()

#Cost vs Scenario
fig1, ax1 = plt.subplots()
plt.scatter(list(range(1,S+1)), cost_SS_s.values(), label="SS")
plt.scatter(list(range(1,S+1)), cost_seq_s.values(), label="Seq")
ax1.set_title('Cost vs Scenario')
ax1.set_xlabel('Scenario')
ax1.set_ylabel('Cost [EUR]') 
ax1.legend(loc='upper left')  
plt.savefig('COst_scen.pdf', dpi=100) 
plt.show()

#Cost as box 
boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
fig2, ax2 = plt.subplots(figsize = (4, 3))
labels = ['Proposed', 'Sequential']
ax2.boxplot([np.array(list(cost_SS_s.values())), np.array(list(cost_seq_s.values()))], labels=labels)
#ax2.boxplot(cost_seq_s.values())
#ax.plot([1,26], [total_cost_int, total_cost_int], color='k', label='Integrated' )
#ax.set_title('Cost vs RMSE')
#ax.set_xlabel('RMSE [EUR/MWh]') 
#ax.set_xlim(5, 25)
#ax.set_ylim(132000, 149000)
scale_y = 1e3
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
ax2.yaxis.set_major_formatter(ticks_y)
ax2.set_ylabel('Cost [kEUR]')
ax2.legend(loc='upper left', frameon=False)   
plt.savefig('Cost_BOX.pdf', dpi=100, bbox_inches='tight')
plt.show()

#Bias vs Scenario
# =============================================================================
# fig2, ax2 = plt.subplots()
# plt.scatter(list(range(1,S+1)), bias_ss.values(), label="SS")
# plt.scatter(list(range(1,S+1)), bias_seq.values(), label="Seq")
# ax2.set_title('Bias vs Scenario')
# ax2.set_xlabel('Scenario')
# ax2.set_ylabel('Bias [EUR/MWh]') 
# ax2.legend(loc='upper left')   
# plt.show()
# =============================================================================
        









