# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:49:52 2022

@author: mikska
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:00:05 2022

@author: mikska
"""
import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
import matplotlib.pyplot as plt
solver = po.SolverFactory('gurobi')
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.weight': 'normal'})

def int_dispatch_function(indata):
    T = indata.T
    #DATA
    time = indata.time 
    gen= indata.gen
    wind= indata.wind
    CHP= indata.CHP
    I= indata.I #Set of dispatchable units
    
    #Technical characteristic
    
    heat_maxprod = indata.heat_maxprod
    elec_maxprod = indata.elec_maxprod
    elec_minprod = indata.elec_minprod
    Ramp = indata.Ramp
    Fuel_min = indata.Fuel_min
    Fuel_max = indata.Fuel_max
    rho_elec = indata.rho_elec # efficiency of the CHP for electricity production
    rho_heat = indata.rho_heat # efficiency of the CHP for heat production
    r_chp = indata.r_chp # elec/heat ratio (flexible in the case of extraction units)
    C_fuel = indata.C_fuel
    C_SU = indata.C_SU
    P_ini = indata.P_ini
    U_ini = indata.U_ini
    
    
    
    D_H =  indata.heat_load #Heat Demand
    D_E = indata.elec_load #El demand
    #L_DA_E_F =indata.Price_DA_actual #Electricity price Day-Ahead forecast
        
        
        
    Wind_DA = indata.Wind_DA  
    
    #MODEL
    integ=pe.ConcreteModel()
    #Duals are desired
    integ.dual = pe.Suffix(direction=pe.Suffix.IMPORT) 
    
    
    ### SETS
    integ.CHP = pe.Set(initialize = CHP) 
    integ.T = pe.Set(initialize = time)
    integ.Tnot1=pe.Set(initialize = time[1:])
    integ.I =pe.Set(initialize = I)
    
    integ.G = pe.Set(initialize = gen ) 
    integ.W = pe.Set(initialize = wind)
    
    
    
    ### PARAMETERS
    integ.Q_max = pe.Param(integ.CHP, initialize = heat_maxprod) #Max integ production
    
    integ.Ramp = pe.Param(integ.I, initialize = Ramp)
    
    integ.Fuel_min = pe.Param(integ.CHP, initialize = Fuel_min)
    integ.Fuel_max = pe.Param(integ.CHP, initialize = Fuel_max)
    integ.rho_elec = pe.Param(integ.CHP, initialize = rho_elec) # efficiency of the CHP for integtricity production
    integ.rho_heat = pe.Param(integ.CHP, initialize = rho_heat) # efficiency of the CHP for integ production
    integ.r_chp = pe.Param(integ.CHP, initialize = r_chp) # el/heat ratio (flexible in the case of extraction units)
    
    integ.C_fuel = pe.Param(integ.I, initialize = C_fuel)
    integ.C_SU = pe.Param(integ.I, initialize = C_SU)
    integ.P_ini = pe.Param(integ.I, initialize = P_ini)
    integ.U_ini = pe.Param(integ.I, initialize = U_ini)
    
    
    
    integ.P_max = pe.Param(integ.G, initialize = elec_maxprod) #Only for Generators
    integ.P_min = pe.Param(integ.G, initialize = elec_minprod) #Only for Generators
    
    integ.D_H = pe.Param(integ.T, initialize = D_H) #integ Demand
    integ.D_E = pe.Param(integ.T, initialize = D_E) #El demand
    
    integ.Wind_DA = pe.Param(integ.W, integ.T, initialize = Wind_DA)
    
    
    ### VARIABLES
    
    integ.q_DA=pe.Var(integ.CHP, integ.T, domain=pe.NonNegativeReals)
    
    integ.p_DA=pe.Var(integ.I, integ.T, domain=pe.NonNegativeReals)
    integ.w_DA=pe.Var(integ.W, integ.T, domain=pe.NonNegativeReals)
    integ.u_DA=pe.Var(integ.I, integ.T, domain=pe.NonNegativeReals)
    integ.c_DA=pe.Var(integ.I, integ.T, domain=pe.NonNegativeReals)
    
    ### CONSTRAINTS
    ##Maximum and Minimum Heat Production
    integ.Heat_DA1 = pe.ConstraintList()
    for t in integ.T:
        for s in integ.CHP:
            integ.Heat_DA1.add( 0 <= integ.q_DA[s,t])
            
    integ.Heat_DA2 = pe.ConstraintList()
    for t in integ.T:
        for s in integ.CHP:
            integ.Heat_DA2.add( integ.q_DA[s,t] <= integ.Q_max[s])
            
    ##CHP operation region
    integ.Heat_DA3 =pe.ConstraintList()
    for t in integ.T:
        for s in integ.CHP:
            integ.Heat_DA3.add( -integ.p_DA[s,t] <= -integ.r_chp[s]*integ.q_DA[s,t])
            
    integ.Heat_DA4 = pe.ConstraintList()
    for t in integ.T:
        for s in integ.CHP:
            integ.Heat_DA4.add( integ.u_DA[s,t]*integ.Fuel_min[s] <= integ.rho_elec[s]*integ.p_DA[s,t]+integ.rho_heat[s]*integ.q_DA[s,t])  
            
    integ.Heat_DA5 = pe.ConstraintList()
    for t in integ.T:
        for s in integ.CHP:
           integ.Heat_DA5.add( integ.rho_elec[s]*integ.p_DA[s,t]+integ.rho_heat[s]*integ.q_DA[s,t] <= integ.u_DA[s,t]*integ.Fuel_max[s])
    
    
    
    ##Maximum and Minimum Wind Production
    integ.El_DA1 = pe.ConstraintList()
    for t in integ.T:
        for w in integ.W:
            integ.El_DA1.add( 0 <= integ.w_DA[w,t])
            
    integ.El_DA2 = pe.ConstraintList()
    for t in integ.T:
        for w in integ.W:
            integ.El_DA2.add( integ.w_DA[w,t] <= integ.Wind_DA[w,t])
    
    #Upper and lower generator bounds
    integ.El_DA3 = pe.ConstraintList()
    for t in integ.T:
        for g in integ.G:
            integ.El_DA3.add( integ.u_DA[g,t]*integ.P_min[g] <= integ.p_DA[g,t])
            
    integ.El_DA4 = pe.ConstraintList()
    for t in integ.T:
        for g in integ.G:
            integ.El_DA4.add( integ.p_DA[g,t] <= integ.u_DA[g,t]*integ.P_max[g])        
    
    
    ##Start-up cost       
    integ.El_DA8 = pe.ConstraintList()
    for t1, t2 in zip(integ.Tnot1, integ.T):
        for i in integ.I:
            integ.El_DA8.add( integ.C_SU[i]*(integ.u_DA[i,t1] - integ.u_DA[i,t2]) <= integ.c_DA[i,t1])
    #t1        
    integ.El_DA9 = pe.ConstraintList()
    for i in integ.I:
        integ.El_DA9.add( integ.C_SU[i]*(integ.u_DA[i,integ.T[1]] - integ.U_ini[i]) <= integ.c_DA[i,integ.T[1]] )
    
    integ.El_DA10 = pe.ConstraintList()
    for i in integ.I:
        for i in integ.I:
            integ.El_DA10.add( 0 <= integ.c_DA[i,t] )
    
    #Relaxed binaries
    integ.El_DA11 = pe.ConstraintList()
    for t in integ.T:
        for i in integ.I:
            integ.El_DA11.add( 0 <= integ.u_DA[i,t] )
    
    integ.El_DA12 = pe.ConstraintList()
    for t in integ.T:
        for i in integ.I:
            integ.El_DA12.add( integ.u_DA[i,t] <= 1 )
            
    #Ramping up amd down CHP
    integ.El_DA13 = pe.ConstraintList()
    for t1, t2 in zip(integ.Tnot1, integ.T):
        for i in integ.I:
            integ.El_DA13.add( integ.p_DA[i,t1] - integ.p_DA[i,t2]  <= integ.Ramp[i]*integ.u_DA[i,t1]) 
    
    integ.El_DA14 = pe.ConstraintList()
    for t1, t2 in zip(integ.Tnot1, integ.T):
        for i in integ.I:
            integ.El_DA14.add( integ.p_DA[i,t2] - integ.p_DA[i,t1]  <= integ.Ramp[i]*integ.u_DA[i,t2])
    #for t1        
    integ.El_DA15 = pe.ConstraintList()
    for i in integ.I:
        integ.El_DA15.add( integ.p_DA[i, integ.T[1]]- integ.P_ini[i] <= integ.Ramp[i]*integ.u_DA[i,integ.T[1]] )
            
    integ.El_DA16 = pe.ConstraintList()
    for i in integ.I:
        integ.El_DA16.add( integ.P_ini[i] - integ.p_DA[i, integ.T[1]]  <= integ.Ramp[i]*integ.U_ini[i] )          
    
    
    #El Balance
    integ.El_DA_bal=pe.ConstraintList()
    for t in integ.T:
        integ.El_DA_bal.add( sum(integ.p_DA[i,t] for i in integ.I) + sum(integ.w_DA[w,t] for w in integ.W)  - integ.D_E[t]  == 0 )
            
    #Heat Balance
    integ.Heat_DA_bal=pe.ConstraintList()
    for t in integ.T:
        integ.Heat_DA_bal.add( sum(integ.q_DA[s,t] for s in integ.CHP) - integ.D_H[t]  == 0 )
            
                       
     ### OBJECTIVE        
    integ.obj=pe.Objective(expr=sum( integ.C_fuel[s]*(integ.rho_heat[s]*integ.q_DA[s,t]+integ.rho_elec[s]*integ.p_DA[s,t]) + integ.c_DA[s,t] for t in integ.T for s in integ.CHP) + sum(integ.C_fuel[g]*integ.p_DA[g,t] + integ.c_DA[g,t] for t in integ.T for g in integ.G ) )
     
    
    
    result=solver.solve(integ, tee=True, symbolic_solver_labels=True)  
    #pe.assert_optimal_termination(result)
    #integ.pprint()
    ##Printing the variables results
    # =============================================================================
    q_CHP={}
    p={}
    for t in integ.T:
         for s in integ.CHP:
             print('q_DA[',t,',',s,',]', pe.value(integ.q_DA[s,t]))
             q_CHP[s,t]=pe.value(integ.q_DA[s,t])
    for t in integ.T:
        for s in integ.CHP:
             print('p_DA[',t,',',i,',]', pe.value(integ.p_DA[i,t]))
             p[s,t]=pe.value(integ.p_DA[s,t])
    # for t in integ.T:
    #     for w in integ.W:
    #         print('w_DA[',t,',',w,',]', pe.value(integ.w_DA[w,t]))
    #for t in integ.T:
    #    for i in integ.I:
    #        print('u_DA[',t,',',i,',]', pe.value(integ.u_DA[i,t]))
    # for t in integ.T:
    #     for i in integ.I:
    #         print('c_DA[',t,',',i,',]', pe.value(integ.c_DA[i,t]))
    # =============================================================================
            
    # =============================================================================
    # ##Dual variables
    # print ("Duals")
    # for c in heat.component_objects(pe.Constraint, active=True):
    #     print (" Constraint",c)
    #     for index in c:
    #         print (" ", index, heat.dual[c[index]])        
    # =============================================================================
    # =============================================================================
    # pe.value(integ.c_DA[i,t])
    # pe.value(integ.q_DA[s,t])
    # pe.value(integ.w_DA[w,t])
    # pe.value(integ.u_DA[i,t])
    # =============================================================================
    
    
    lambda_DA_H={}        
    #print ("Heat_prices")
    for index, t in zip(integ.Heat_DA_bal, integ.T):    
            lambda_DA_H[t] = integ.dual[integ.Heat_DA_bal[index]]
    #print(lambda_DA_H)
        
    
    lambda_DA_E={}        
   # print ("Electricity prices")
    for index, t in zip(integ.El_DA_bal, integ.T):    
            lambda_DA_E[t] = integ.dual[integ.El_DA_bal[index]]
    #print(lambda_DA_E)
        
    print('Obj_IN=',pe.value(integ.obj))
    fig1, ax1 = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(list(lambda_DA_E.values()))
    
    total_cost_it={}
    
    for t in integ.T:
        for s in integ.CHP:
            total_cost_it[s,t]=integ.C_fuel[s]*(integ.rho_heat[s]*pe.value(integ.q_DA[s,t])+integ.rho_elec[s]*pe.value(integ.p_DA[s,t])) + pe.value(integ.c_DA[s,t])        
    for t in integ.T:
        for g in integ.G:
            total_cost_it[g,t]=integ.C_fuel[g]*pe.value(integ.p_DA[g,t]) + pe.value(integ.c_DA[g,t])        
    #print(total_cost_it)
    
    total_cost2 = sum(total_cost_it[s,t] for t in integ.T for s in integ.CHP) + sum(total_cost_it[g,t] for t in integ.T for g in integ.G)
    print("Total_cost2= ", total_cost2)
    total_cost1=sum( integ.C_fuel[s]*(integ.rho_heat[s]*pe.value(integ.q_DA[s,t]) + integ.rho_elec[s]*pe.value(integ.p_DA[s,t])) + pe.value(integ.c_DA[s,t]) for t in integ.T for s in integ.CHP ) + sum(integ.C_fuel[g]*pe.value(integ.p_DA[g,t]) + pe.value(integ.c_DA[g,t]) for t in integ.T for g in integ.G)
    print("Total_cost1= ", total_cost1)
    #total_cost1= sum(t, sum(s,    + c_DA.l(s,t)) + sum(g, idata(g,'C_fuel')*p_DA.l(g,t) +c_DA.l(g,t)  )   );
    cost_heat=sum( integ.C_fuel[s]*(integ.rho_heat[s]*pe.value(integ.q_DA[s,t])) + pe.value(integ.c_DA[s,t]) for t in integ.T for s in integ.CHP ) 
    print("Cost_heat= ", cost_heat)
    cost_elec=sum( integ.C_fuel[s]*(integ.rho_elec[s]*pe.value(integ.p_DA[s,t]))  for t in integ.T for s in integ.CHP ) + sum(integ.C_fuel[g]*pe.value(integ.p_DA[g,t]) + pe.value(integ.c_DA[g,t]) for t in integ.T for g in integ.G)
    print("Cost_elec= ", cost_elec)
    profit_it={}
    
    for t in integ.T:
        for s in integ.CHP:
            profit_it[s,t]= pe.value(integ.p_DA[s,t])*(lambda_DA_E[t] - integ.rho_elec[s]*integ.C_fuel[s]) + pe.value(integ.q_DA[s,t])*(lambda_DA_H[t] - integ.rho_heat[s]*integ.C_fuel[s]) - pe.value(integ.c_DA[s,t])        
    for t in integ.T:
        for g in integ.G:
            profit_it[g,t]= pe.value(integ.p_DA[g,t])*(lambda_DA_E[t] - integ.C_fuel[g]) - pe.value(integ.c_DA[g,t])        
    #print(profit_it)
    profit={}
    for i in integ.I:
        profit[i]=sum(profit_it[i,t] for t in integ.T)
    print('Profit of each unit')
    print(profit)
    
    #Wind curtailemnt
    Wind_curt={}
    for t in integ.T:
        for w in integ.W:
            Wind_curt[w,t]=integ.Wind_DA[w,t] - pe.value(integ.w_DA[w,t])   
    Wind_curt_total=sum(Wind_curt[w,t] for t in integ.T for w in integ.W)
    #Heat dispatch Integrated
    fig, ax = plt.subplots(figsize = (4, 3))
    q_chp={}
    for s in integ.CHP:
        q_t=[]
        for t in integ.T:
            q_t.append(pe.value(integ.q_DA[s,t]))
        q_chp[s]=q_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    
  
    ax.stackplot(indata.time_numbers, q_chp.values(),
             labels=q_chp.keys(), alpha=0.8)
    #ax.set_title('Heat dispatch: Integrated')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Heat [MW]') 
    ax.set_ylim(0, 600)
    #ax.legend(loc='upper left')   
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)
    ax.legend(bbox_to_anchor=(1.17, 1.05), fancybox=True, shadow=True)   
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    H_DA={}
    for i in q_chp.keys():
        H_DA[i]=int(sum(q_chp[i]))

    ax.text(4, 140, H_DA['CHP1'])
    ax.text(4, 340, H_DA['CHP2'])

    plt.savefig('Heat_int.pdf', dpi=100, bbox_inches='tight')
    plt.show    
    
    #Electricity dispatch Integrated
    fig1, ax1 = plt.subplots(figsize = (4, 3))
    p_DA_all={}
    wind_SP=['W_SP']
    for i in integ.I:
        p_t=[]
        for t in integ.T:
            p_t.append(pe.value(integ.p_DA[i,t]))
        p_DA_all[i]=p_t
    for w in integ.W:
        p_t=[]
        for t in integ.T:
            p_t.append(pe.value(integ.w_DA[w,t]))
        p_DA_all[w]=p_t
    for w, w1 in zip(integ.W, wind_SP):
        p_t=[]
        for t in integ.T:
            p_t.append(integ.Wind_DA[w,t] - pe.value(integ.w_DA[w,t])   )
        p_DA_all[w1]=p_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    labels1=[str(), str(), 'G1', 'G2', 'W1', 'Wind spillage']    
    ax1.stackplot(indata.time_numbers, p_DA_all.values(),
                 labels=labels1, alpha=0.8)
    #ax1.set_title('Power dispatch: Integrated')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Power [MW]') 
    ax1.set_ylim(0, 900)
    #ax1.legend(loc='upper left')  
   # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
   #             mode="expand", borderaxespad=0, ncol=3)
    ax1.legend(bbox_to_anchor=(1.17, 1.05), fancybox=True, shadow=True)   
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    #plt.xticks(np.arange(min(indata.time_numbers)+1, max(indata.time_numbers)+1, 3.0))
    E_DA={}
    for i in p_DA_all.keys():
        E_DA[i]=int(sum(p_DA_all[i]))
    ax1.text(8, 100, E_DA['CHP1'])
    ax1.text(8, 340, E_DA['CHP2'])
    ax1.text(8, 540, E_DA['G2'])
    ax1.text(8, 720, E_DA['W1'])
    ax1.text(3, 400, E_DA['W_SP'])
    plt.savefig('Power_int.pdf', dpi=100, bbox_inches='tight')
    plt.show    
    #Dispatch and price
    fig4, ax4 = plt.subplots(figsize = (4, 3))
    #color = 'b'
    ax4.set_xlabel('Time Period')
    ax4.set_ylabel('Power [MW]')
    ax4.bar(indata.time_numbers,p_DA_all['CHP2'], fill=False, edgecolor='tab:blue', label='CHP2')
    #ax4.legend()
    ax4.set_ylim(0, 280)
    ax5 = ax4.twinx()
    #color = 'k'
    ax5.plot(list(lambda_DA_E.values()), color='k', label='Electricity Price')
    ax5.set_ylim(0, 80)
    ax5.set_ylabel('Price [EUR/MWh]')
    #ax5.legend()
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax5.get_legend_handles_labels()
    ax4.legend(h1+h2, l1+l2, frameon=False)
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    fig4.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Power_Price_int.pdf', dpi=100, bbox_inches='tight')
    plt.show()
    
    #Plot CHP disptach
    x=[0 ,   76.92, 300, 300, 0];
    y=[47.6, 38.46, 150, 250, 285.7];
    fig7, ax7 = plt.subplots(figsize = (4, 3))
    plt.plot(x,y)
    plt.scatter(q_CHP['CHP1','t9'], p['CHP1','t9'],  label="CHP1,t9", facecolors='none', edgecolors='r')
    plt.annotate("CHP1,t9", (q_CHP['CHP1','t9']-52, p['CHP1','t9']+8))
    #plt.scatter(q_CHP['CHP1','t2'], p['CHP1','t2'],  label="CHP1,t2", facecolors='none', edgecolors='b')
    #plt.annotate("CHP1,t2", (q_CHP['CHP1','t2']-52, p['CHP1','t2']+8))
    #plt.scatter(q_CHP['CHP1','t3'], p['CHP1','t3'],  label="CHP1,t3", facecolors='none', edgecolors='g')
    #plt.annotate("CHP1,t3", (q_CHP['CHP1','t3']+2, p['CHP1','t3']-10))
    plt.scatter(q_CHP['CHP2','t9'], p['CHP2','t9'],  label="CHP2,t9", facecolors='none', edgecolors='c')
    plt.annotate("CHP2,t9", (q_CHP['CHP2','t9']+2, p['CHP2','t9']+10))
    #plt.scatter(q_CHP['CHP2','t2'], p['CHP2','t2'],  label="CHP2,t2", facecolors='none', edgecolors='m')
    #plt.annotate("CHP2,t2", (q_CHP['CHP2','t2']+2, p['CHP2','t2']+2))
    #plt.scatter(q_CHP['CHP2','t3'], p['CHP2','t3'],  label="CHP3,t3", facecolors='none', edgecolors='k')
    #plt.annotate("CHP2,t3", (q_CHP['CHP2','t3']+2, p['CHP2','t3']))
    ax7.set_xlabel('Heat [MW]')
    ax7.set_ylabel('Power [MW]') 
    #ax.set_title('CHP dispatch: Sequential')
    #ax.legend(loc='lower right')   
    ax7.set_ylim(-10, 300)
    plt.savefig('CHP_dispatch_int_big.pdf', dpi=100, bbox_inches='tight')
    plt.show
    u_DA_all={}
    for s in integ.CHP:
        u_t=[]
        for t in integ.T:
            u_t.append(pe.value(integ.u_DA[s,t]))
        u_DA_all[s]=u_t
    for g in integ.G:
        u_t=[]
        for t in integ.T:
            u_t.append(pe.value(integ.u_DA[g,t]))
            u_DA_all[g]=u_t
# =============================================================================
#     fig8, ax8 = plt.subplots(figsize = (4, 3))
#     plt.scatter(indata.time_numbers,u_DA_all['CHP1'], label='CHP1')
#     fig9, ax9 = plt.subplots(figsize = (4, 3))
#     plt.scatter(indata.time_numbers,u_DA_all['CHP2'], label='CHP2')
#     fig10, ax10 = plt.subplots(figsize = (4, 3))
#     plt.scatter(indata.time_numbers,u_DA_all['G1'], label='G1')
#     fig11, ax11 = plt.subplots(figsize = (4, 3))
#     plt.scatter(indata.time_numbers,u_DA_all['G2'], label='G2')
# =============================================================================
    # =============================================================================
    # y_w={}    
    # for w in range(0,S):
    #     y_t=[]
    #     for t in range(0,T):
    #         #print('t= ', t, 'w= ', w, 'y=' , y['t{0}'.format(t), 'w{0}'.format(w)])
    #         y_t.append(y[t, w])            
    #     #print(y_t)
    #     y_w['w{0}'.format(w)]=y_t
    #     plt.plot(y_w['w{0}'.format(w)])
    # 
    # =============================================================================
    return Wind_curt_total, total_cost1, lambda_DA_E, lambda_DA_H



