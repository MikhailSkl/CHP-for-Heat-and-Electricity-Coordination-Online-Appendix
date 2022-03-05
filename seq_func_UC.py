# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:59:40 2022

@author: mikska
"""
import pyomo.environ as pe
import pyomo.opt as po
#import data_initialization4 as indata
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.weight': 'normal'})
def seq_dispatch_func_UC(indata, L_DA_E_F):
    solver = po.SolverFactory('gurobi')
    np.random.seed(886)
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
    heat=pe.ConcreteModel()
    #Duals are desired
    heat.dual = pe.Suffix(direction=pe.Suffix.IMPORT) 
    
    
    ### SETS
    heat.CHP = pe.Set(initialize = CHP) 
    heat.T = pe.Set(initialize = time)
    heat.Tnot1=pe.Set(initialize = time[1:])
    heat.I =pe.Set(initialize = I)
    
    ### PARAMETERS
    heat.heat_maxprod = pe.Param(heat.CHP, initialize = heat_maxprod) #Max heat production
    heat.Ramp = pe.Param(heat.I, initialize = Ramp)
    heat.Fuel_min = pe.Param(heat.CHP, initialize = Fuel_min)
    heat.Fuel_max = pe.Param(heat.CHP, initialize = Fuel_max)
    heat.rho_elec = pe.Param(heat.CHP, initialize = rho_elec) # efficiency of the CHP for electricity production
    heat.rho_heat = pe.Param(heat.CHP, initialize = rho_heat) # efficiency of the CHP for heat production
    heat.r_chp = pe.Param(heat.CHP, initialize = r_chp) # elec/heat ratio (flexible in the case of extraction units)
    heat.C_fuel = pe.Param(heat.I, initialize = C_fuel)
    heat.C_SU = pe.Param(heat.I, initialize = C_SU)
    heat.P_ini = pe.Param(heat.I, initialize = P_ini)
    heat.U_ini = pe.Param(heat.I, initialize = U_ini)
    
    heat.D_H = pe.Param(heat.T, initialize = D_H) #Heat Demand
    heat.L_DA_E_F = pe.Param(heat.T, initialize = L_DA_E_F) #Electricity price Day-Ahead forecast
    
    
    ### VARIABLES
    
    heat.q_DA=pe.Var(heat.CHP, heat.T, domain=pe.NonNegativeReals)
    heat.p_DA_H=pe.Var(heat.CHP, heat.T, domain=pe.NonNegativeReals)
    heat.u_DA=pe.Var(heat.CHP, heat.T, domain=pe.Binary)
    heat.c_DA=pe.Var(heat.CHP, heat.T, domain=pe.NonNegativeReals)
    
    ### CONSTRAINTS
    
    ##Maximum and Minimum Heat Production
    
    heat.Heat_DA1 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA1.add( 0 <= heat.q_DA[s,t])
            
    heat.Heat_DA2 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA2.add( heat.q_DA[s,t] <= heat.heat_maxprod[s])
            
    ##CHP operation region
    
    heat.Heat_DA3 =pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA3.add( -heat.p_DA_H[s,t] <= -heat.r_chp[s]*heat.q_DA[s,t])
            
    heat.Heat_DA4 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA4.add( heat.u_DA[s,t]*heat.Fuel_min[s] <= heat.rho_elec[s]*heat.p_DA_H[s,t]+heat.rho_heat[s]*heat.q_DA[s,t])  
            
    heat.Heat_DA5 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA5.add( heat.rho_elec[s]*heat.p_DA_H[s,t]+heat.rho_heat[s]*heat.q_DA[s,t] <= heat.u_DA[s,t]*heat.Fuel_max[s])
    
    ##Start-up cost       
    heat.Heat_DA6 = pe.ConstraintList()
    for t1, t2 in zip(heat.Tnot1, heat.T):
        for s in heat.CHP:
            heat.Heat_DA6.add( heat.C_SU[s]*(heat.u_DA[s,t1] - heat.u_DA[s,t2]) <= heat.c_DA[s,t1])
    #for t1        
    heat.Heat_DA7 = pe.ConstraintList()
    for s in heat.CHP:
        heat.Heat_DA7.add( heat.C_SU[s]*(heat.u_DA[s,heat.T[1]] - heat.U_ini[s]) <= heat.c_DA[s,heat.T[1]] )
    
    heat.Heat_DA8 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA8.add( 0 <= heat.c_DA[s,t] )
    
    #Relaxed binaries
    heat.Heat_DA9 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA9.add( 0 <= heat.u_DA[s,t] )
    
    heat.Heat_DA10 = pe.ConstraintList()
    for t in heat.T:
        for s in heat.CHP:
            heat.Heat_DA10.add( heat.u_DA[s,t] <= 1 )
            
    #Ramping up amd down
    heat.Heat_DA11 = pe.ConstraintList()
    for t1, t2 in zip(heat.Tnot1, heat.T):
        for s in heat.CHP:
            heat.Heat_DA11.add( heat.p_DA_H[s,t1] - heat.p_DA_H[s,t2]  <= heat.Ramp[s]*heat.u_DA[s,t1]) 
    
    heat.Heat_DA12 = pe.ConstraintList()
    for t1, t2 in zip(heat.Tnot1, heat.T):
        for s in heat.CHP:
            heat.Heat_DA12.add( heat.p_DA_H[s,t2] - heat.p_DA_H[s,t1]  <= heat.Ramp[s]*heat.u_DA[s,t2])
    #for t1        
    heat.Heat_DA13 = pe.ConstraintList()
    for s in heat.CHP:
        heat.Heat_DA13.add( heat.p_DA_H[s, heat.T[1]]- heat.P_ini[s] <= heat.Ramp[s]*heat.u_DA[s,heat.T[1]] )
            
    heat.Heat_DA14 = pe.ConstraintList()
    for s in heat.CHP:
        heat.Heat_DA14.add( heat.P_ini[s] - heat.p_DA_H[s, heat.T[1]]  <= heat.Ramp[s]*heat.U_ini[s] )        
    
    #Heat Balance
    heat.Heat_DA_bal=pe.ConstraintList()
    for t in heat.T:
        heat.Heat_DA_bal.add( sum(heat.q_DA[s,t] for s in heat.CHP) - heat.D_H[t]  == 0 )
            
                       
    
    ### OBJECTIVE        
    heat.obj=pe.Objective(expr=sum( heat.C_fuel[s]*(heat.rho_heat[s]*heat.q_DA[s,t]+heat.rho_elec[s]*heat.p_DA_H[s,t]) + heat.c_DA[s,t] - L_DA_E_F[t]*heat.p_DA_H[s,t] for t in heat.T for s in heat.CHP))
     
    
    
    result_h=solver.solve(heat, tee=True, symbolic_solver_labels=True)  
    #pe.assert_optimal_termination(result)
    #heat.pprint()
    ##Printing the variables results
    q_CHP={}
    # =============================================================================
    for t in heat.T:
         for s in heat.CHP:
             #print('q_DA[',t,',',s,',]', pe.value(heat.q_DA[s,t]))
             q_CHP[s,t]=pe.value(heat.q_DA[s,t])
    # for t in heat.T:
    #     for s in heat.CHP:
    #         print('p_DA_H[',t,',',s,',]', pe.value(heat.p_DA_H[s,t]))
    # for t in heat.T:
    #     for s in heat.CHP:
    #         print('u_DA[',t,',',s,',]', pe.value(heat.u_DA[s,t]))
    # =============================================================================
            
    # =============================================================================
    # ##Dual variables
    # print ("Duals")
    # for c in heat.component_objects(pe.Constraint, active=True):
    #     print (" Constraint",c)
    #     for index in c:
    #         print (" ", index, heat.dual[c[index]])        
    # =============================================================================
    
    #lambda_DA_H={}        
    #print ("Heat_prices")
    #for index, t in zip(heat.Heat_DA_bal, heat.T):    
    #        lambda_DA_H[t] = heat.dual[heat.Heat_DA_bal[index]]
    #print(lambda_DA_H)
        
    print('Obj=',pe.value(heat.obj))
    
    q_DA_fix={}
    u_DA_fix={}
    ##Fixing values
    for t in heat.T:
        for s in heat.CHP:
            q_DA_fix[s,t]=pe.value(heat.q_DA[s,t])        
    for t in heat.T:
        for s in heat.CHP:
            u_DA_fix[s,t]=pe.value(heat.u_DA[s,t])
    #print(q_DA_fix)
    #print(u_DA_fix)
    p_chp_exp={}
    for s in heat.CHP:
        p_t=[]
        for t in heat.T:
            p_t.append(pe.value(heat.p_DA_H[s,t]))
        p_chp_exp[s]=p_t
         
    ##DEFINING ELECTRICITY DISPATCH
    print ("Sequential Electricity Dispatch")
    #MODEL
    elec=pe.ConcreteModel()
    #Duals are desired
    elec.dual = pe.Suffix(direction=pe.Suffix.IMPORT) 
    
    
    ### SETS
    elec.CHP = pe.Set(initialize = CHP) 
    elec.G = pe.Set(initialize = gen ) 
    elec.W = pe.Set(initialize = wind)
    elec.T = pe.Set(initialize = time)
    elec.Tnot1=pe.Set(initialize = time[1:])
    elec.I =pe.Set(initialize = I)
    
    ### PARAMETERS
    elec.P_max = pe.Param(elec.G, initialize = elec_maxprod) #Only for Generators
    elec.P_min = pe.Param(elec.G, initialize = elec_minprod) #Only for Generators
    elec.Ramp = pe.Param(elec.I, initialize = Ramp)
    elec.Fuel_min = pe.Param(elec.CHP, initialize = Fuel_min)
    elec.Fuel_max = pe.Param(elec.CHP, initialize = Fuel_max)
    elec.rho_elec = pe.Param(elec.CHP, initialize = rho_elec) # efficiency of the CHP for electricity production
    elec.rho_heat = pe.Param(elec.CHP, initialize = rho_heat) # efficiency of the CHP for heat production
    elec.r_chp = pe.Param(elec.CHP, initialize = r_chp) # elec/heat ratio (flexible in the case of extraction units)
    elec.C_fuel = pe.Param(elec.I, initialize = C_fuel)
    elec.C_SU = pe.Param(elec.I, initialize = C_SU)
    elec.P_ini = pe.Param(elec.I, initialize = P_ini)
    elec.U_ini = pe.Param(elec.I, initialize = U_ini)
    
    elec.D_E = pe.Param(elec.T, initialize = D_E) #El demand
    elec.L_DA_E_F = pe.Param(elec.T, initialize = L_DA_E_F) #Electricity price Day-Ahead forecast
    
    elec.Wind_DA = pe.Param(elec.W, elec.T, initialize = Wind_DA)
    
    #Fixed parameters from heat dispatch
    elec.q_DA_fix = pe.Param(elec.CHP, elec.T, initialize = q_DA_fix)
    elec.u_DA_fix = pe.Param(elec.CHP, elec.T, initialize = u_DA_fix)
    
    
    ### VARIABLES
    elec.p_DA=pe.Var(elec.I, elec.T, domain=pe.NonNegativeReals)
    elec.w_DA=pe.Var(elec.W, elec.T, domain=pe.NonNegativeReals)
    elec.u_DA=pe.Var(elec.G, elec.T, domain=pe.Binary)
    elec.c_DA=pe.Var(elec.G, elec.T, domain=pe.NonNegativeReals)
    
    ### CONSTRAINTS
    
    ##Maximum and Minimum Wind Production
    
    elec.El_DA1 = pe.ConstraintList()
    for t in elec.T:
        for w in elec.W:
            elec.El_DA1.add( 0 <= elec.w_DA[w,t])
            
    elec.El_DA2 = pe.ConstraintList()
    for t in elec.T:
        for w in elec.W:
            elec.El_DA2.add( elec.w_DA[w,t] <= elec.Wind_DA[w,t])
    
    #Upper and lower generator bounds
    elec.El_DA3 = pe.ConstraintList()
    for t in elec.T:
        for g in elec.G:
            elec.El_DA3.add( elec.u_DA[g,t]*elec.P_min[g] <= elec.p_DA[g,t])
            
    elec.El_DA4 = pe.ConstraintList()
    for t in elec.T:
        for g in elec.G:
            elec.El_DA4.add( elec.p_DA[g,t] <= elec.u_DA[g,t]*elec.P_max[g])        
    
    ##CHP operation region
    
    elec.El_DA5 =pe.ConstraintList()
    for t in elec.T:
        for s in elec.CHP:
            elec.El_DA5.add( -elec.p_DA[s,t] <= -elec.r_chp[s]*elec.q_DA_fix[s,t])
            
    elec.El_DA6 = pe.ConstraintList()
    for t in elec.T:
        for s in elec.CHP:
            elec.El_DA6.add( elec.u_DA_fix[s,t]*elec.Fuel_min[s] <= elec.rho_elec[s]*elec.p_DA[s,t]+elec.rho_heat[s]*elec.q_DA_fix[s,t])  
            
    elec.El_DA7 = pe.ConstraintList()
    for t in elec.T:
        for s in elec.CHP:
            elec.El_DA7.add( elec.rho_elec[s]*elec.p_DA[s,t]+elec.rho_heat[s]*elec.q_DA_fix[s,t] <= elec.u_DA_fix[s,t]*elec.Fuel_max[s])
    
    ##Start-up cost       
    elec.El_DA8 = pe.ConstraintList()
    for t1, t2 in zip(elec.Tnot1, elec.T):
        for g in elec.G:
            elec.El_DA8.add( elec.C_SU[g]*(elec.u_DA[g,t1] - elec.u_DA[g,t2]) <= elec.c_DA[g,t1])
    #t1        
    elec.El_DA9 = pe.ConstraintList()
    for g in elec.G:
        elec.El_DA9.add( elec.C_SU[g]*(elec.u_DA[g,elec.T[1]] - elec.U_ini[g]) <= elec.c_DA[g,elec.T[1]] )
    
    elec.El_DA10 = pe.ConstraintList()
    for t in elec.T:
        for g in elec.G:
            elec.El_DA10.add( 0 <= elec.c_DA[g,t] )
    
    #Relaxed binaries
    elec.El_DA11 = pe.ConstraintList()
    for t in elec.T:
        for g in elec.G:
            elec.El_DA11.add( 0 <= elec.u_DA[g,t] )
    
    elec.El_DA12 = pe.ConstraintList()
    for t in elec.T:
        for g in elec.G:
            elec.El_DA12.add( elec.u_DA[g,t] <= 1 )
            
    #Ramping up amd down CHP
    elec.El_DA13 = pe.ConstraintList()
    for t1, t2 in zip(elec.Tnot1, elec.T):
        for s in elec.CHP:
            elec.El_DA13.add( elec.p_DA[s,t1] - elec.p_DA[s,t2]  <= elec.Ramp[s]*elec.u_DA_fix[s,t1]) 
    
    elec.El_DA14 = pe.ConstraintList()
    for t1, t2 in zip(elec.Tnot1, elec.T):
        for s in elec.CHP:
            elec.El_DA14.add( elec.p_DA[s,t2] - elec.p_DA[s,t1]  <= elec.Ramp[s]*elec.u_DA_fix[s,t2])
    #for t1        
    elec.El_DA15 = pe.ConstraintList()
    for s in elec.CHP:
        elec.El_DA15.add( elec.p_DA[s, elec.T[1]]- elec.P_ini[s] <= elec.Ramp[s]*elec.u_DA_fix[s,elec.T[1]] )
            
    elec.El_DA16 = pe.ConstraintList()
    for s in elec.CHP:
        elec.El_DA16.add( elec.P_ini[s] - elec.p_DA[s, elec.T[1]]  <= elec.Ramp[s]*elec.U_ini[s] )          
    
    #Ramping up amd down G
    elec.El_DA17 = pe.ConstraintList()
    for t1, t2 in zip(elec.Tnot1, elec.T):
        for g in elec.G:
            elec.El_DA17.add( elec.p_DA[g,t1] - elec.p_DA[g,t2]  <= elec.Ramp[g]*elec.u_DA[g,t1]) 
    
    elec.El_DA18 = pe.ConstraintList()
    for t1, t2 in zip(elec.Tnot1, elec.T):
        for g in elec.G:
            elec.El_DA18.add( elec.p_DA[g,t2] - elec.p_DA[g,t1]  <= elec.Ramp[g]*elec.u_DA[g,t2])
    #for t1        
    elec.El_DA19 = pe.ConstraintList()
    for g in elec.G:
        elec.El_DA19.add( elec.p_DA[g, elec.T[1]]- elec.P_ini[g] <= elec.Ramp[g]*elec.u_DA[g,elec.T[1]] )
            
    elec.El_DA20 = pe.ConstraintList()
    for g in elec.G:
        elec.El_DA20.add( elec.P_ini[g] - elec.p_DA[g, elec.T[1]]  <= elec.Ramp[g]*elec.U_ini[g] ) 
    
    #El Balance
    elec.El_DA_bal=pe.ConstraintList()
    for t in elec.T:
        elec.El_DA_bal.add( sum(elec.p_DA[i,t] for i in elec.I) + sum(elec.w_DA[w,t] for w in elec.W)  - elec.D_E[t]  == 0 )
            
                       
    
    ### OBJECTIVE        
    elec.obj=pe.Objective(expr=sum( elec.C_fuel[s]*(elec.rho_elec[s]*elec.p_DA[s,t]) for t in elec.T for s in elec.CHP) + sum(elec.C_fuel[g]*elec.p_DA[g,t] + elec.c_DA[g,t] for t in elec.T for g in elec.G ) )
     
    
    
    result_e=solver.solve(elec, tee=True, symbolic_solver_labels=True)  
    #pe.assert_optimal_termination(result)
    #elec.pprint()
    ##Printing the variables results
    # =============================================================================
# =============================================================================
#     for t in heat.T:
#         for s in heat.CHP:
#             print('q_DA[',t,',',s,',]', pe.value(heat.q_DA[s,t]))
#     for t in heat.T:
#         for s in heat.CHP:
#             print('p_DA_H[',t,',',s,',]', pe.value(heat.p_DA_H[s,t]))        
# =============================================================================
    p={}
    for t in elec.T:
        for s in elec.CHP:
           # print('p_DA[',t,',',i,',]', pe.value(elec.p_DA[i,t]))
            p[s,t]=pe.value(elec.p_DA[s,t])
    # for t in elec.T:
    #     for w in elec.W:
    #         print('w_DA[',t,',',w,',]', pe.value(elec.w_DA[w,t]))
    # =============================================================================
    
    #lambda_DA_E_seq={}        
    #print ("Electricity prices")
    #for index, t in zip(elec.El_DA_bal, elec.T):    
    #        lambda_DA_E_seq[t] = elec.dual[elec.El_DA_bal[index]]
    #print(lambda_DA_E_seq)
        
    print('Obj_EL=',pe.value(elec.obj))
    
    total_cost_it={}
    
    for t in heat.T:
        for s in heat.CHP:
            total_cost_it[s,t]=heat.C_fuel[s]*(heat.rho_heat[s]*pe.value(heat.q_DA[s,t])+elec.rho_elec[s]*pe.value(elec.p_DA[s,t])) + pe.value(heat.c_DA[s,t])        
    for t in elec.T:
        for g in elec.G:
            total_cost_it[g,t]=elec.C_fuel[g]*pe.value(elec.p_DA[g,t]) + pe.value(elec.c_DA[g,t])        
    #print(total_cost_it)
    
    total_cost2 = sum(total_cost_it[s,t] for t in heat.T for s in heat.CHP) + sum(total_cost_it[g,t] for t in elec.T for g in elec.G)
    print("Total_cost_UC2= ", total_cost2)
    total_cost1=sum( elec.C_fuel[s]*(heat.rho_heat[s]*pe.value(heat.q_DA[s,t]) + elec.rho_elec[s]*pe.value(elec.p_DA[s,t])) + pe.value(heat.c_DA[s,t]) for t in heat.T for s in elec.CHP ) + sum(elec.C_fuel[g]*pe.value(elec.p_DA[g,t]) + pe.value(elec.c_DA[g,t]) for t in elec.T for g in elec.G)
    print("Total_cost_UC1= ", total_cost1)
    #total_cost1= sum(t, sum(s,    + c_DA.l(s,t)) + sum(g, idata(g,'C_fuel')*p_DA.l(g,t) +c_DA.l(g,t)  )   );
    cost_heat=sum( heat.C_fuel[s]*(heat.rho_heat[s]*pe.value(heat.q_DA[s,t])) + pe.value(heat.c_DA[s,t]) for t in heat.T for s in heat.CHP ) 
    print("Cost_heat= ", cost_heat)
    cost_elec=sum( elec.C_fuel[s]*(elec.rho_elec[s]*pe.value(elec.p_DA[s,t]))  for t in elec.T for s in elec.CHP ) + sum(elec.C_fuel[g]*pe.value(elec.p_DA[g,t]) + pe.value(elec.c_DA[g,t]) for t in elec.T for g in elec.G)
    print("Cost_elec= ", cost_elec)
    profit_it={}
    profit_it={}
    
   # for t in heat.T:
   #     for s in heat.CHP:
   #         profit_it[s,t]= pe.value(elec.p_DA[s,t])*(lambda_DA_E_seq[t] - heat.rho_elec[s]*heat.C_fuel[s]) + pe.value(heat.q_DA[s,t])*(lambda_DA_H[t] - heat.rho_heat[s]*heat.C_fuel[s]) - pe.value(heat.c_DA[s,t])        
   # for t in elec.T:
    #    for g in elec.G:
    #        profit_it[g,t]= pe.value(elec.p_DA[g,t])*(lambda_DA_E_seq[t] - elec.C_fuel[g]) - pe.value(elec.c_DA[g,t])        
    #print(profit_it)
    profit={}
    #for i in elec.I:
    #    profit[i]=sum(profit_it[i,t] for t in elec.T)
    print('Profit of each unit')
    print(profit)
    
    
    #Wind curtailemnt
    Wind_curt={}
    for t in elec.T:
        for w in elec.W:
            Wind_curt[w,t]=elec.Wind_DA[w,t] - pe.value(elec.w_DA[w,t])   
    Wind_curt_total=sum(Wind_curt[w,t] for t in elec.T for w in elec.W)
    
       
    fig, ax = plt.subplots(figsize = (4, 3))
    q_chp={}
    for s in heat.CHP:
        q_t=[]
        for t in heat.T:
            q_t.append(pe.value(heat.q_DA[s,t]))
        q_chp[s]=q_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    
    ax.stackplot(indata.time_numbers, q_chp.values(),
                 labels=q_chp.keys(), alpha=0.8)
    #ax.set_title('Heat dispatch: Sequential')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Heat [MW]') 
    ax.set_ylim(0, 600)
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)
    ax.legend(bbox_to_anchor=(1.17, 1.05), fancybox=True, shadow=True)   
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    #ax.legend(loc='upper left')  
    H_DA={}
    for i in q_chp.keys():
        H_DA[i]=int(sum(q_chp[i]))

    ax.text(4, 140, H_DA['CHP1'])
    ax.text(4, 340, H_DA['CHP2'])
    plt.savefig('Heat_seq_UC.pdf', dpi=100, bbox_inches='tight')
    plt.show    
    #Electricity dispatch Integrated
    
    p_DA_all={}
    wind_SP=['W_SP']
    for i in elec.I:
        p_t=[]
        for t in elec.T:
            p_t.append(pe.value(elec.p_DA[i,t]))
        p_DA_all[i]=p_t
    for w in elec.W:
        p_t=[]
        for t in elec.T:
            p_t.append(pe.value(elec.w_DA[w,t]))
        p_DA_all[w]=p_t
    for w, w1 in zip(elec.W, wind_SP):
        p_t=[]
        for t in elec.T:
            p_t.append(elec.Wind_DA[w,t] - pe.value(elec.w_DA[w,t])   )
        p_DA_all[w1]=p_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    fig1, ax1 = plt.subplots(figsize = (4, 3))
    labels1=[str(), str(), 'G1', 'G2', 'W1', 'Wind spillage']
    ax1.stackplot(indata.time_numbers, p_DA_all.values(),
                 labels=labels1, alpha=0.8)
    #ax1.set_title('Power dispatch: Sequential')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Power [MW]') 
    ax1.set_ylim(0, 900)
    #ax1.legend(loc='upper left') 
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)
    ax1.legend(bbox_to_anchor=(1.17, 1.05), fancybox=True, shadow=True)   
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    E_DA={}
    for i in p_DA_all.keys():
        E_DA[i]=int(sum(p_DA_all[i]))
    ax1.text(8, 100, E_DA['CHP1'])
    ax1.text(8, 280, E_DA['CHP2'])
    ax1.text(8, 385, E_DA['G1'])
    ax1.text(8, 540, E_DA['G2'])
    ax1.text(8, 720, E_DA['W1'])
    ax1.text(3, 400, E_DA['W_SP'])
    plt.savefig('Power_seq_UC.pdf', dpi=100, bbox_inches='tight')
    plt.show  
    
    
    #Dispatch and price
# =============================================================================
#     fig4, ax4 = plt.subplots(figsize = (4, 3))
#     #color = 'b'
#     ax4.set_xlabel('Time Period')
#     ax4.set_ylabel('Power [MW]')
#     ax4.bar(indata.time_numbers,p_DA_all['CHP2'], fill=False, edgecolor='tab:blue', label='CHP2')
#     #ax4.legend()
#     ax4.set_ylim(0, 280)
#     ax5 = ax4.twinx()
#     #color = 'k'
#     ax5.plot(list(lambda_DA_E_seq.values()), color='k', label='Electricity Price')
#     ax5.set_ylim(0, 80)
#     ax5.set_ylabel('Price [EUR/MWh]')
#     #ax5.legend()
#     h1, l1 = ax4.get_legend_handles_labels()
#     h2, l2 = ax5.get_legend_handles_labels()
#     ax4.legend(h1+h2, l1+l2, loc='upper right', frameon=False)
#     plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
#     fig4.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.savefig('Power_Price_seq.pdf', dpi=100, bbox_inches='tight')
#     plt.show()
# =============================================================================
    
       #Expected dispatch and price
    fig6, ax6 = plt.subplots(figsize = (4, 3))
    #color = 'b'
    ax6.set_xlabel('Time Period')
    ax6.set_ylabel('Power [MW]')
    ax6.bar(indata.time_numbers,p_chp_exp['CHP2'], fill=False, edgecolor='tab:red', label='CHP2')
    #ax4.legend()
    ax6.set_ylim(0, 280)
    ax7 = ax6.twinx()
    #color = 'k'
    ax7.plot(list(L_DA_E_F.values()), color='k', label='Forecasted Price')
    ax7.set_ylim(0, 80)
    ax7.set_ylabel('Price [EUR/MWh]')
    #ax5.legend()
    h1, l1 = ax6.get_legend_handles_labels()
    h2, l2 = ax7.get_legend_handles_labels()
    ax6.legend(h1+h2, l1+l2, loc='upper left', frameon=False)
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    #ax6.legend(bbox_to_anchor=(1.17, 1.05), loc="lower left" )
    fig6.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Exp_Power_Estim_Price_seq_UC.pdf', dpi=100, bbox_inches='tight')
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
    plt.savefig('CHP_dispatch_seq_big_UC.pdf', dpi=100, bbox_inches='tight')
    plt.show
    
    
    return Wind_curt_total, total_cost1





