# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:42:52 2022

@author: mikska
"""
import pyomo.environ as pe
import pyomo.opt as po
import pyomo.mpec as mpc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.weight': 'normal'})


def SS_dispatch_func(indata, L_DA_E_F, scen):   
    np.random.seed(886)
    
    T = indata.T
    CH = indata.CH
    G = indata.G
    W = indata.W
    #DATA
    time = indata.time 
    gen= indata.gen
    wind= indata.wind
    CHP= indata.CHP
    I = indata.I #Set of dispatchable units
    
    SSC=indata.SSC
    #Non self-scheduling CHPs
    NSS=indata.NSS
    #Non-self scheduling units
    N=indata.N
    
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
               
    Wind_DA = indata.Wind_DA  #Wind power forecast
    
    #MODEL
    SS_NLP=pe.ConcreteModel()
    #Duals are desired
    #SS_ dual = pe.Suffix(direction=pe.Suffix.IMPORT) 
    
    
    ### SETS
    SS_NLP.CHP = pe.Set(initialize = CHP) 
    SS_NLP.T = pe.Set(initialize = time)
    SS_NLP.Tnot1=pe.Set(initialize = time[1:])
    SS_NLP.I =pe.Set(initialize = I)
    
    SS_NLP.G = pe.Set(initialize = gen ) 
    SS_NLP.W = pe.Set(initialize = wind)
    
    SS_NLP.SSC = pe.Set(initialize = SSC ) 
    SS_NLP.NSS = pe.Set(initialize = NSS )
    
    SS_NLP.N = pe.Set(initialize = N )
    
    
    ### PARAMETERS
    SS_NLP.Q_max = pe.Param(SS_NLP.CHP, initialize = heat_maxprod) #Max SS production
    
    SS_NLP.Ramp = pe.Param(SS_NLP.I, initialize = Ramp)
    
    SS_NLP.Fuel_min = pe.Param(SS_NLP.CHP, initialize = Fuel_min)
    SS_NLP.Fuel_max = pe.Param(SS_NLP.CHP, initialize = Fuel_max)
    SS_NLP.rho_elec = pe.Param(SS_NLP.CHP, initialize = rho_elec) # efficiency of the CHP for SStricity production
    SS_NLP.rho_heat = pe.Param(SS_NLP.CHP, initialize = rho_heat) # efficiency of the CHP for SS production
    SS_NLP.r_chp = pe.Param(SS_NLP.CHP, initialize = r_chp) # el/heat ratio (flexible in the case of extraction units)
    
    SS_NLP.C_fuel = pe.Param(SS_NLP.I, initialize = C_fuel)
    SS_NLP.C_SU = pe.Param(SS_NLP.I, initialize = C_SU)
    SS_NLP.P_ini = pe.Param(SS_NLP.I, initialize = P_ini)
    SS_NLP.U_ini = pe.Param(SS_NLP.I, initialize = U_ini)
    
    
    
    SS_NLP.P_max = pe.Param(SS_NLP.G, initialize = elec_maxprod) #Only for Generators
    SS_NLP.P_min = pe.Param(SS_NLP.G, initialize = elec_minprod) #Only for Generators
    
    SS_NLP.D_H = pe.Param(SS_NLP.T, initialize = D_H) #SS Demand
    SS_NLP.D_E = pe.Param(SS_NLP.T, initialize = D_E) #El demand
    
    SS_NLP.Wind_DA = pe.Param(SS_NLP.W, SS_NLP.T, initialize = Wind_DA)
    
    SS_NLP.L_DA_E_F = pe.Param(SS_NLP.T, initialize = L_DA_E_F) #Electricity price Day-Ahead forecast
    
    
    
    ### VARIABLES
    def _bounds_rule_q(m, chp, t):
        return (0, heat_maxprod[chp])
    SS_NLP.q_DA=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=_bounds_rule_q)
    
    SS_NLP.p_DA_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals)
    
    SS_NLP.p_DA=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    def _bounds_rule_w(m, j, t):
        return (0, Wind_DA[j,t])
    ##def _init_rule_W_DA(m, j, t):
     #   return (Wind_DA[j,t])
    SS_NLP.w_DA=pe.Var(SS_NLP.W, SS_NLP.T, domain=pe.NonNegativeReals, bounds=_bounds_rule_w)
    
    SS_NLP.u_DA=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0,1))
    
    #def _bounds_rule_c(m, j, t):
    #    return (0, C_SU[j])
    SS_NLP.c_DA=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    
    ##Dual variables
    #positive variables
    SS_NLP.mu_min_H=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_H=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_C_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_C_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_C_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_R_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_R_H=pe.Var(SS_NLP.NSS, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_C=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_C=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_C=pe.Var(SS_NLP.CHP, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_W=pe.Var(SS_NLP.W, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_max_W=pe.Var(SS_NLP.W, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    
    SS_NLP.mu_min_G=pe.Var(SS_NLP.G, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_G=pe.Var(SS_NLP.G, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_SU=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_SU=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_B=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_B=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    SS_NLP.mu_min_R=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    SS_NLP.mu_max_R=pe.Var(SS_NLP.I, SS_NLP.T, domain=pe.NonNegativeReals, bounds=(0, None))
    
    
    #free variables
    SS_NLP.lambda_DA_E=pe.Var(SS_NLP.T, domain=pe.NonNegativeReals) # day-ahead electricity price in period t [$ per MWh]

    SS_NLP.lambda_DA_H=pe.Var(SS_NLP.T, domain=pe.NonNegativeReals) #day-ahead heat price in period t [$ per MWh]
    ##KKT HEAT MARKET
    ##Stationarity conditions
            
    SS_NLP.L1_q_DA = pe.ConstraintList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.L1_q_DA.add( SS_NLP.C_fuel[s]*SS_NLP.rho_heat[s] - SS_NLP.mu_min_H[s,t] + SS_NLP.mu_max_H[s,t] - SS_NLP.lambda_DA_H[t] + SS_NLP.mu_C_H[s,t]*SS_NLP.r_chp[s] - SS_NLP.mu_min_C_H[s,t]*SS_NLP.rho_heat[s] + SS_NLP.mu_max_C_H[s,t]*SS_NLP.rho_heat[s] == 0)       
    
    SS_NLP.L1_p_DA = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for s in SS_NLP.NSS:
            SS_NLP.L1_p_DA.add( SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s] - SS_NLP.L_DA_E_F[t1] - SS_NLP.mu_C_H[s,t1] - SS_NLP.mu_min_C_H[s,t1]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C_H[s,t1]*SS_NLP.rho_elec[s] \
                            + SS_NLP.mu_max_R_H[s,t1]  - SS_NLP.mu_max_R_H[s,t2] + SS_NLP.mu_min_R_H[s,t2] - SS_NLP.mu_min_R_H[s,t1] == 0 )       
    #t24
    SS_NLP.L1_p_DA1 = pe.ConstraintList()
    for s in SS_NLP.NSS:
        SS_NLP.L1_p_DA1.add( SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s] - SS_NLP.L_DA_E_F[SS_NLP.T[T]] - SS_NLP.mu_C_H[s,SS_NLP.T[T]] - SS_NLP.mu_min_C_H[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C_H[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] \
                        + SS_NLP.mu_max_R_H[s,SS_NLP.T[T]] - SS_NLP.mu_min_R_H[s,SS_NLP.T[T]] == 0)       
    
    SS_NLP.L1_u_DA = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for s in SS_NLP.NSS:
            SS_NLP.L1_u_DA.add( SS_NLP.mu_min_C_H[s,t1]*SS_NLP.Fuel_min[s] - SS_NLP.mu_max_C_H[s,t1]*SS_NLP.Fuel_max[s] + SS_NLP.mu_max_SU[s,t1]*SS_NLP.C_SU[s] - SS_NLP.mu_max_SU[s,t2]*SS_NLP.C_SU[s] - SS_NLP.mu_min_B[s,t1] + SS_NLP.mu_max_B[s,t1] \
                           - SS_NLP.mu_max_R_H[s,t1]*SS_NLP.Ramp[s] - SS_NLP.mu_min_R_H[s,t2]*SS_NLP.Ramp[s] == 0 )
    #t24
    SS_NLP.L1_u_DA1 = pe.ConstraintList()
    for s in SS_NLP.NSS:
        SS_NLP.L1_u_DA1.add( SS_NLP.mu_min_C_H[s,SS_NLP.T[T]]*SS_NLP.Fuel_min[s] - SS_NLP.mu_max_C_H[s,SS_NLP.T[T]]*SS_NLP.Fuel_max[s] + SS_NLP.mu_max_SU[s,SS_NLP.T[T]]*SS_NLP.C_SU[s] - SS_NLP.mu_min_B[s,SS_NLP.T[T]] + SS_NLP.mu_max_B[s,SS_NLP.T[T]] \
                       - SS_NLP.mu_max_R_H[s,SS_NLP.T[T]]*SS_NLP.Ramp[s] == 0 )
            
    SS_NLP.L1_c_DA = pe.ConstraintList()   
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.L1_c_DA.add( 1 - SS_NLP.mu_max_SU[s,t] - SS_NLP.mu_min_SU[s,t] == 0 )
            
    
    ##Complementarity Constraints
    
    SS_NLP.comp1 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp1.add(expr=mpc.complements( SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_min_H[s,t] >= 0))
            
    SS_NLP.comp2 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.comp2.add(expr=mpc.complements( SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_max_H[s,t] >= 0))
    
    SS_NLP.comp3 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp3.add(expr=mpc.complements( SS_NLP.p_DA_H[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_C_H[s,t] >= 0))
            
    SS_NLP.comp4 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.comp4.add(expr=mpc.complements( SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] >= 0, SS_NLP.mu_min_C_H[s,t] >= 0))
    
    SS_NLP.comp5 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp5.add(expr=mpc.complements( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t]>= 0, SS_NLP.mu_max_C_H[s,t] >= 0))
            
    SS_NLP.comp6 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            SS_NLP.comp6.add(expr=mpc.complements( SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) >= 0, SS_NLP.mu_max_SU[s,t1] >= 0))
          
    SS_NLP.comp7 = mpc.ComplementarityList()     
    for s in SS_NLP.NSS:
        SS_NLP.comp7.add(expr=mpc.complements( SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) >= 0, SS_NLP.mu_max_SU[s,SS_NLP.T[1]] >= 0))
             
    SS_NLP.comp8 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp8.add(expr=mpc.complements( SS_NLP.c_DA[s,t] >= 0, SS_NLP.mu_min_SU[s,t] >= 0))
         
    SS_NLP.comp9 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            SS_NLP.comp9.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA_H[s,t2] - SS_NLP.p_DA_H[s,t1] >=0, SS_NLP.mu_max_R_H[s,t1] >= 0))
                         
    SS_NLP.comp10 = mpc.ComplementarityList()     
    for s in SS_NLP.NSS:
        SS_NLP.comp10.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA_H[s,SS_NLP.T[1]] >=0, SS_NLP.mu_max_R_H[s,SS_NLP.T[1]] >= 0))
    
    SS_NLP.comp11 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            SS_NLP.comp11.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA_H[s,t1] - SS_NLP.p_DA_H[s,t2] >=0, SS_NLP.mu_min_R_H[s,t1] >=0))
                         
    SS_NLP.comp12 = mpc.ComplementarityList()     
    for s in SS_NLP.NSS:
        SS_NLP.comp12.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA_H[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] >= 0, SS_NLP.mu_min_R_H[s,SS_NLP.T[1]] >=0))
    
    SS_NLP.comp13 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp13.add(expr=mpc.complements( SS_NLP.u_DA[s,t] >= 0, SS_NLP.mu_min_B[s,t] >= 0))
            
    SS_NLP.comp14 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.comp14.add(expr=mpc.complements( 1 - SS_NLP.u_DA[s,t] >= 0, SS_NLP.mu_max_B[s,t] >= 0))
    
    ##KKT ELECTRICITY MARKET   
    ##Stationarity conditions
                 
    SS_NLP.L2_p_DAs = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for s in SS_NLP.NSS:
            SS_NLP.L2_p_DAs.add( SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s] - SS_NLP.lambda_DA_E[t1] - SS_NLP.mu_C[s,t1] - SS_NLP.mu_min_C[s,t1]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C[s,t1]*SS_NLP.rho_elec[s] \
                            + SS_NLP.mu_max_R[s,t1]  - SS_NLP.mu_max_R[s,t2] + SS_NLP.mu_min_R[s,t2] - SS_NLP.mu_min_R[s,t1] == 0 )
                
    #t24
    SS_NLP.L2_p_DAs1 = pe.ConstraintList()
    for s in SS_NLP.NSS:
        SS_NLP.L2_p_DAs1.add( SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s] - SS_NLP.lambda_DA_E[SS_NLP.T[T]] - SS_NLP.mu_C[s,SS_NLP.T[T]] - SS_NLP.mu_min_C[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] \
                        + SS_NLP.mu_max_R[s,SS_NLP.T[T]] - SS_NLP.mu_min_R[s,SS_NLP.T[T]] == 0)  
            
    
    SS_NLP.L2_p_DAg = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for g in SS_NLP.G:
            SS_NLP.L2_p_DAg.add( SS_NLP.C_fuel[g] - SS_NLP.lambda_DA_E[t1] - SS_NLP.mu_min_G[g,t1] + SS_NLP.mu_max_G[g,t1] \
                            + SS_NLP.mu_max_R[g,t1]  - SS_NLP.mu_max_R[g,t2] + SS_NLP.mu_min_R[g,t2] - SS_NLP.mu_min_R[g,t1] == 0 )    
                
    #t24
    SS_NLP.L2_p_DAg1 = pe.ConstraintList()
    for g in SS_NLP.G:
        SS_NLP.L2_p_DAg1.add( SS_NLP.C_fuel[g] - SS_NLP.lambda_DA_E[SS_NLP.T[T]]  - SS_NLP.mu_min_G[g,SS_NLP.T[T]] + SS_NLP.mu_max_G[g,SS_NLP.T[T]] \
                        + SS_NLP.mu_max_R[g,SS_NLP.T[T]] - SS_NLP.mu_min_R[g,SS_NLP.T[T]] == 0)  
                                                     
    
    SS_NLP.L2_u_DAg = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for g in SS_NLP.G:
            SS_NLP.L2_u_DAg.add( SS_NLP.mu_min_G[g,t1]*SS_NLP.P_min[g] - SS_NLP.mu_max_G[g,t1]*SS_NLP.P_max[g] + SS_NLP.mu_max_SU[g,t1]*SS_NLP.C_SU[g] - SS_NLP.mu_max_SU[g,t2]*SS_NLP.C_SU[g] - SS_NLP.mu_min_B[g,t1] + SS_NLP.mu_max_B[g,t1] \
                           - SS_NLP.mu_max_R[g,t1]*SS_NLP.Ramp[g] - SS_NLP.mu_min_R[g,t2]*SS_NLP.Ramp[g] == 0 )
                           
    #t24
    SS_NLP.L2_u_DAg1 = pe.ConstraintList()
    for g in SS_NLP.G:
        SS_NLP.L2_u_DAg1.add( SS_NLP.mu_min_G[g,SS_NLP.T[T]]*SS_NLP.P_min[g] - SS_NLP.mu_max_G[g,SS_NLP.T[T]]*SS_NLP.P_max[g] + SS_NLP.mu_max_SU[g,SS_NLP.T[T]]*SS_NLP.C_SU[g] - SS_NLP.mu_min_B[g,SS_NLP.T[T]] + SS_NLP.mu_max_B[g,SS_NLP.T[T]] \
                       - SS_NLP.mu_max_R[g,SS_NLP.T[T]]*SS_NLP.Ramp[g] == 0 )
            
            
            
    SS_NLP.L2_w_DA = pe.ConstraintList()
    for t in SS_NLP.T:
        for w in SS_NLP.W:
            SS_NLP.L2_w_DA.add( -SS_NLP.lambda_DA_E[t] - SS_NLP.mu_min_W[w,t] + SS_NLP.mu_max_W[w,t] == 0)
        
    SS_NLP.L2_c_DA = pe.ConstraintList()   
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            SS_NLP.L2_c_DA.add( 1 - SS_NLP.mu_max_SU[g,t] - SS_NLP.mu_min_SU[g,t] == 0 )        
        
    ##Complementarity conditions
    SS_NLP.comp15 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for w in SS_NLP.W:
           SS_NLP.comp15.add(expr=mpc.complements( SS_NLP.w_DA[w,t] >= 0, SS_NLP.mu_min_W[w,t] >= 0))
            
    SS_NLP.comp16 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for w in SS_NLP.W:
            SS_NLP.comp16.add(expr=mpc.complements( SS_NLP.Wind_DA[w,t] - SS_NLP.w_DA[w,t] >= 0, SS_NLP.mu_max_W[w,t] >= 0))
    
    SS_NLP.comp17 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for g in SS_NLP.G:
           SS_NLP.comp17.add(expr=mpc.complements( SS_NLP.p_DA[g,t] - SS_NLP.P_min[g]*SS_NLP.u_DA[g,t] >= 0, SS_NLP.mu_min_G[g,t] >= 0))
            
    SS_NLP.comp18 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            SS_NLP.comp18.add(expr=mpc.complements( SS_NLP.P_max[g]*SS_NLP.u_DA[g,t] - SS_NLP.p_DA[g,t] >= 0, SS_NLP.mu_max_G[g,t] >= 0))
    
    SS_NLP.comp19 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp19.add(expr=mpc.complements( SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_C[s,t] >= 0))
            
    SS_NLP.comp20 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            SS_NLP.comp20.add(expr=mpc.complements( SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] >= 0, SS_NLP.mu_min_C[s,t] >= 0))
    
    SS_NLP.comp21 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
           SS_NLP.comp21.add(expr=mpc.complements( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] - SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_max_C[s,t] >= 0))
    
    SS_NLP.comp22 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for g in SS_NLP.G:
            SS_NLP.comp22.add(expr=mpc.complements( SS_NLP.c_DA[g,t1] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,t1] - SS_NLP.u_DA[g,t2]) >= 0, SS_NLP.mu_max_SU[g,t1] >= 0))
          
    SS_NLP.comp23 = mpc.ComplementarityList()     
    for g in SS_NLP.G:
        SS_NLP.comp23.add(expr=mpc.complements( SS_NLP.c_DA[g,SS_NLP.T[1]] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,SS_NLP.T[1]] - SS_NLP.U_ini[g]) >= 0, SS_NLP.mu_max_SU[g,SS_NLP.T[1]] >= 0))
             
    SS_NLP.comp24 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for g in SS_NLP.G:
           SS_NLP.comp24.add(expr=mpc.complements( SS_NLP.c_DA[g,t] >= 0, SS_NLP.mu_min_SU[g,t] >= 0))
    
    
    
    SS_NLP.comp25 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for n in SS_NLP.N:
            SS_NLP.comp25.add(expr=mpc.complements( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t1] + SS_NLP.p_DA[n,t2] - SS_NLP.p_DA[n,t1] >=0, SS_NLP.mu_max_R[n,t1] >= 0))
     
                        
    SS_NLP.comp26 = mpc.ComplementarityList()     
    for n in SS_NLP.N:
        SS_NLP.comp26.add(expr=mpc.complements( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,SS_NLP.T[1]] + SS_NLP.P_ini[n] - SS_NLP.p_DA[n,SS_NLP.T[1]] >=0, SS_NLP.mu_max_R[n,SS_NLP.T[1]] >= 0))
    
    SS_NLP.comp27 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for n in SS_NLP.N:
            SS_NLP.comp27.add(expr=mpc.complements( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t2] + SS_NLP.p_DA[n,t1] - SS_NLP.p_DA[n,t2] >=0, SS_NLP.mu_min_R[n,t1] >=0))
                         
    SS_NLP.comp28 = mpc.ComplementarityList()     
    for n in SS_NLP.N:
        SS_NLP.comp28.add(expr=mpc.complements( SS_NLP.Ramp[n]*SS_NLP.U_ini[n] + SS_NLP.p_DA[n,SS_NLP.T[1]] - SS_NLP.P_ini[n] >=0, SS_NLP.mu_min_R[n,SS_NLP.T[1]] >=0))
    
    
    
    
    SS_NLP.comp29 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for g in SS_NLP.G:
           SS_NLP.comp29.add(expr=mpc.complements( SS_NLP.u_DA[g,t] >= 0, SS_NLP.mu_min_B[g,t] >= 0))
            
    SS_NLP.comp30 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            SS_NLP.comp30.add(expr=mpc.complements( 1 - SS_NLP.u_DA[g,t] >= 0, SS_NLP.mu_max_B[g,t] >= 0))
            
    
    ##KKT Self-scheduling  
    ##Stationarity conditions
    SS_NLP.L3_q_DA = pe.ConstraintList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            SS_NLP.L3_q_DA.add( - SS_NLP.lambda_DA_H[t] + SS_NLP.C_fuel[s]*SS_NLP.rho_heat[s] - SS_NLP.mu_min_H[s,t] + SS_NLP.mu_max_H[s,t]  + SS_NLP.mu_C[s,t]*SS_NLP.r_chp[s] - SS_NLP.mu_min_C[s,t]*SS_NLP.rho_heat[s] + SS_NLP.mu_max_C[s,t]*SS_NLP.rho_heat[s] == 0)       
    
    SS_NLP.L3_p_DA = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for s in SS_NLP.SSC:
            SS_NLP.L3_p_DA.add( - SS_NLP.lambda_DA_E[t1] + SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s]  - SS_NLP.mu_C[s,t1] - SS_NLP.mu_min_C[s,t1]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C[s,t1]*SS_NLP.rho_elec[s] \
                            + SS_NLP.mu_max_R[s,t1]  - SS_NLP.mu_max_R[s,t2] + SS_NLP.mu_min_R[s,t2] - SS_NLP.mu_min_R[s,t1] == 0 )   
                            
    #t24
    SS_NLP.L3_p_DA1 = pe.ConstraintList()
    for s in SS_NLP.SSC:
        SS_NLP.L3_p_DA1.add( - SS_NLP.lambda_DA_E[SS_NLP.T[T]] + SS_NLP.C_fuel[s]*SS_NLP.rho_elec[s]  - SS_NLP.mu_C[s,SS_NLP.T[T]] - SS_NLP.mu_min_C[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] + SS_NLP.mu_max_C[s,SS_NLP.T[T]]*SS_NLP.rho_elec[s] \
                        + SS_NLP.mu_max_R[s,SS_NLP.T[T]] - SS_NLP.mu_min_R[s,SS_NLP.T[T]] == 0)   
                                         
    SS_NLP.L3_u_DA = pe.ConstraintList()
    for t1, t2 in zip(SS_NLP.T, SS_NLP.Tnot1):
        for s in SS_NLP.SSC:
            SS_NLP.L3_u_DA.add( SS_NLP.mu_min_C[s,t1]*SS_NLP.Fuel_min[s] - SS_NLP.mu_max_C[s,t1]*SS_NLP.Fuel_max[s] + SS_NLP.mu_max_SU[s,t1]*SS_NLP.C_SU[s] - SS_NLP.mu_max_SU[s,t2]*SS_NLP.C_SU[s] - SS_NLP.mu_min_B[s,t1] + SS_NLP.mu_max_B[s,t1] \
                           - SS_NLP.mu_max_R[s,t1]*SS_NLP.Ramp[s] - SS_NLP.mu_min_R[s,t2]*SS_NLP.Ramp[s] == 0 )
                                       
    #t24
    SS_NLP.L3_u_DA1 = pe.ConstraintList()
    for s in SS_NLP.SSC:
        SS_NLP.L3_u_DA1.add( SS_NLP.mu_min_C[s,SS_NLP.T[T]]*SS_NLP.Fuel_min[s] - SS_NLP.mu_max_C[s,SS_NLP.T[T]]*SS_NLP.Fuel_max[s] + SS_NLP.mu_max_SU[s,SS_NLP.T[T]]*SS_NLP.C_SU[s] - SS_NLP.mu_min_B[s,SS_NLP.T[T]] + SS_NLP.mu_max_B[s,SS_NLP.T[T]] \
                       - SS_NLP.mu_max_R[s,SS_NLP.T[T]]*SS_NLP.Ramp[s] == 0 )
                   
    SS_NLP.L3_c_DA = pe.ConstraintList()   
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            SS_NLP.L3_c_DA.add( 1 - SS_NLP.mu_max_SU[s,t] - SS_NLP.mu_min_SU[s,t] == 0 )
            
    ##Complementarity conditions        
    SS_NLP.comp31 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
           SS_NLP.comp31.add(expr=mpc.complements( SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_min_H[s,t] >= 0))
            
    SS_NLP.comp32 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            SS_NLP.comp32.add(expr=mpc.complements( SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_max_H[s,t] >= 0))
    
    SS_NLP.comp33 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
           SS_NLP.comp33.add(expr=mpc.complements( SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] >= 0, SS_NLP.mu_C[s,t] >= 0))
            
    SS_NLP.comp34 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            SS_NLP.comp34.add(expr=mpc.complements( SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] >= 0, SS_NLP.mu_min_C[s,t] >= 0))
    
    SS_NLP.comp35 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
           SS_NLP.comp35.add(expr=mpc.complements( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t]>= 0, SS_NLP.mu_max_C[s,t] >= 0))
    
    SS_NLP.comp36 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            SS_NLP.comp36.add(expr=mpc.complements( SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) >= 0, SS_NLP.mu_max_SU[s,t1] >= 0))
          
    SS_NLP.comp37 = mpc.ComplementarityList()     
    for s in SS_NLP.SSC:
        SS_NLP.comp37.add(expr=mpc.complements( SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) >= 0, SS_NLP.mu_max_SU[s,SS_NLP.T[1]] >= 0))
             
    SS_NLP.comp38 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
           SS_NLP.comp38.add(expr=mpc.complements( SS_NLP.c_DA[s,t] >= 0, SS_NLP.mu_min_SU[s,t] >= 0))
    
    SS_NLP.comp39 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            SS_NLP.comp39.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA[s,t2] - SS_NLP.p_DA[s,t1] >=0, SS_NLP.mu_max_R[s,t1] >= 0))
                         
    SS_NLP.comp40 = mpc.ComplementarityList()     
    for s in SS_NLP.SSC:
        SS_NLP.comp40.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA[s,SS_NLP.T[1]] >=0, SS_NLP.mu_max_R[s,SS_NLP.T[1]] >= 0))
    
    SS_NLP.comp41 = mpc.ComplementarityList()     
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            SS_NLP.comp41.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA[s,t1] - SS_NLP.p_DA[s,t2] >=0, SS_NLP.mu_min_R[s,t1] >=0))
                         
    SS_NLP.comp42 = mpc.ComplementarityList()     
    for s in SS_NLP.SSC:
        SS_NLP.comp42.add(expr=mpc.complements( SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] >=0, SS_NLP.mu_min_R[s,SS_NLP.T[1]] >=0))
    
    
    SS_NLP.comp43 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
           SS_NLP.comp43.add(expr=mpc.complements( SS_NLP.u_DA[s,t] >= 0, SS_NLP.mu_min_B[s,t] >= 0))
            
    SS_NLP.comp44 = mpc.ComplementarityList()
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            SS_NLP.comp44.add(expr=mpc.complements( 1 - SS_NLP.u_DA[s,t] >= 0, SS_NLP.mu_max_B[s,t] >= 0))
    
                
    #El Balance
    SS_NLP.El_DA_bal=pe.ConstraintList()
    for t in SS_NLP.T:
        SS_NLP.El_DA_bal.add( sum(SS_NLP.p_DA[i,t] for i in SS_NLP.I) + sum(SS_NLP.w_DA[w,t] for w in SS_NLP.W)  - SS_NLP.D_E[t]  == 0 )
           
    #Heat Balance
    SS_NLP.Heat_DA_bal=pe.ConstraintList()
    for t in SS_NLP.T:
        SS_NLP.Heat_DA_bal.add( sum(SS_NLP.q_DA[s,t] for s in SS_NLP.CHP) - SS_NLP.D_H[t]  == 0 )
        
     ### OBJECTIVE        
    #SS_NLP.obj=pe.Objective(expr=1)
    
    #Solve
    solver = po.SolverFactory('mpec_nlp')
    solver.options['epsilon_initial']=1e+4
    #solver.options['epsilon_initial']=1e+4
    if scen=='s49':
       solver.options['epsilon_initial']=1e+5
    if scen=='s29':
       solver.options['epsilon_initial']=1e+5
    solver.options['epsilon_final']=1e-2
    ##For some scenarious it takes 4-5 hours to solve the problem. therefore we need more precision
    if scen=='s1':
       solver.options['epsilon_final']=1e-3
    if scen=='s6':
       solver.options['epsilon_final']=1e-2
    if scen=='s5':
       solver.options['epsilon_final']=1e-3
    if scen=='s9':
       solver.options['epsilon_final']=1e-3
    if scen=='s10':
       solver.options['epsilon_final']=1e-3
    if scen=='s11':
       solver.options['epsilon_final']=1e-3
    if scen=='s14':
       solver.options['epsilon_final']=1e-3
    if scen=='s23':
       solver.options['epsilon_final']=1e-3
    if scen=='s24':
       solver.options['epsilon_final']=1e-3
    if scen=='s25':
       solver.options['epsilon_final']=1e-3
    if scen=='s28':
       solver.options['epsilon_final']=1e-4
    if scen=='s32':
       solver.options['epsilon_final']=1e-3
    if scen=='s38':
       solver.options['epsilon_final']=1e-1
    if scen=='s39':
       solver.options['epsilon_final']=1e-1
    if scen=='s40':
       solver.options['epsilon_final']=1e-3
    if scen=='s41':
       solver.options['epsilon_final']=1e-1
    if scen=='s42':
       solver.options['epsilon_final']=1e-3
    if scen=='s47':
       solver.options['epsilon_final']=1e-3
    print("\n NLP Displaying Solution NLP\n" + '-'*60)
    results = solver.solve(SS_NLP, load_solutions=True)
    print("\n NLP Displaying Solution NLP\n" + '-'*60)
    #sends results to stdout
    results.write()
    
    ##Set Big-M variables
    M_tune=1.8
    M_c=0
    #comp1    
    comp_p1=np.array([pe.value(SS_NLP.q_DA[s,t]) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value(SS_NLP.mu_min_H[s,t]) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp1=M_tune*max(comp_p1)+M_c
    M2_comp1=M_tune*max(comp_p2)+M_c
    #comp2
    comp_p1=np.array([pe.value(SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t]) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value(SS_NLP.mu_max_H[s,t]) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp2=M_tune*max(comp_p1)+M_c
    M2_comp2=M_tune*max(comp_p2)+M_c
    #comp3
    comp_p1=np.array([pe.value( SS_NLP.p_DA_H[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_C_H[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp3=M_tune*max(comp_p1)+M_c
    M2_comp3=M_tune*max(comp_p2)+M_c    
    #comp4
    comp_p1=np.array([pe.value( SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_C_H[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp4=M_tune*max(comp_p1)+M_c
    M2_comp4=M_tune*max(comp_p2)+M_c         
    #comp5
    comp_p1=np.array([pe.value( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_C_H[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp5=M_tune*max(comp_p1)+M_c
    M2_comp5=M_tune*max(comp_p2)+M_c   
    #comp6
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    M1_comp6=M_tune*max(comp_p1)+M_c
    M2_comp6=M_tune*max(comp_p2)+M_c 
    #comp7
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) ) for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[s,SS_NLP.T[1]] ) for s in SS_NLP.NSS ])
    M1_comp7=M_tune*max(comp_p1)+M_c
    M2_comp7=M_tune*max(comp_p2)+M_c 
    #comp8
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_SU[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp8=M_tune*max(comp_p1)+M_c
    M2_comp8=M_tune*max(comp_p2)+M_c 
    #comp9
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA_H[s,t2] - SS_NLP.p_DA_H[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R_H[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    M1_comp9=M_tune*max(comp_p1)+M_c
    M2_comp9=M_tune*max(comp_p2)+M_c 
    #comp10
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA_H[s,SS_NLP.T[1]] ) for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R_H[s,SS_NLP.T[1]] ) for s in SS_NLP.NSS ])
    M1_comp10=M_tune*max(comp_p1)+M_c
    M2_comp10=M_tune*max(comp_p2)+M_c 
    #comp11
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA_H[s,t1] - SS_NLP.p_DA_H[s,t2] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R_H[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.NSS ])
    M1_comp11=M_tune*max(comp_p1)+M_c
    M2_comp11=M_tune*max(comp_p2)+M_c 
    #comp12
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA_H[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] )  for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R_H[s,SS_NLP.T[1]] )  for s in SS_NLP.NSS ])
    M1_comp12=M_tune*max(comp_p1)+M_c
    M2_comp12=M_tune*max(comp_p2)+M_c 
    #comp13
    comp_p1=np.array([pe.value( SS_NLP.u_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_B[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp13=M_tune*max(comp_p1)+M_c
    M2_comp13=M_tune*max(comp_p2)+M_c 
    #comp14
    comp_p1=np.array([pe.value( 1 - SS_NLP.u_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_B[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp14=M_tune*max(comp_p1)+M_c
    M2_comp14=M_tune*max(comp_p2)+M_c 
    #comp15
    comp_p1=np.array([pe.value( SS_NLP.w_DA[w,t] ) for t in SS_NLP.T for w in SS_NLP.W ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_W[w,t] ) for t in SS_NLP.T for w in SS_NLP.W])
    M1_comp15=M_tune*max(comp_p1)+M_c
    M2_comp15=M_tune*max(comp_p2)+M_c 
    #comp16
    comp_p1=np.array([pe.value( SS_NLP.Wind_DA[w,t] - SS_NLP.w_DA[w,t] ) for t in SS_NLP.T for w in SS_NLP.W ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_W[w,t] ) for t in SS_NLP.T for w in SS_NLP.W ])
    M1_comp16=M_tune*max(comp_p1)+M_c
    M2_comp16=M_tune*max(comp_p2)+M_c 
    #comp17
    comp_p1=np.array([pe.value( SS_NLP.p_DA[g,t] - SS_NLP.P_min[g]*SS_NLP.u_DA[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_G[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    M1_comp17=M_tune*max(comp_p1)+M_c
    M2_comp17=M_tune*max(comp_p2)+M_c 
    #comp18
    comp_p1=np.array([pe.value( SS_NLP.P_max[g]*SS_NLP.u_DA[g,t] - SS_NLP.p_DA[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_G[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    M1_comp18=M_tune*max(comp_p1)+M_c
    M2_comp18=M_tune*max(comp_p2)+M_c 
    #comp19
    comp_p1=np.array([pe.value( SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp19=M_tune*max(comp_p1)+M_c
    M2_comp19=M_tune*max(comp_p2)+M_c 
    #comp20
    comp_p1=np.array([pe.value( SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp20=M_tune*max(comp_p1)+M_c
    M2_comp20=M_tune*max(comp_p2)+M_c 
    #comp21
    comp_p1=np.array([pe.value( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] - SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.NSS ])
    M1_comp21=M_tune*max(comp_p1)+M_c
    M2_comp21=M_tune*max(comp_p2)+M_c 
    #comp22
    comp_p1=np.array([pe.value( SS_NLP.c_DA[g,t1] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,t1] - SS_NLP.u_DA[g,t2]) ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[g,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for g in SS_NLP.G ])
    M1_comp22=M_tune*max(comp_p1)+M_c
    M2_comp22=M_tune*max(comp_p2)+M_c 
    #comp23
    comp_p1=np.array([pe.value( SS_NLP.c_DA[g,SS_NLP.T[1]] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,SS_NLP.T[1]] - SS_NLP.U_ini[g]) )  for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[g,SS_NLP.T[1]] ) for g in SS_NLP.G ])
    M1_comp23=M_tune*max(comp_p1)+M_c
    M2_comp23=M_tune*max(comp_p2)+M_c 
    #comp24
    comp_p1=np.array([pe.value( SS_NLP.c_DA[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_SU[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    M1_comp24=M_tune*max(comp_p1)+M_c
    M2_comp24=M_tune*max(comp_p2)+M_c 
    #comp25
    comp_p1=np.array([pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t1] + SS_NLP.p_DA[n,t2] - SS_NLP.p_DA[n,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for n in SS_NLP.N ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R[n,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for n in SS_NLP.N ])
    M1_comp25=M_tune*max(comp_p1)+M_c
    M2_comp25=M_tune*max(comp_p2)+M_c 
    #comp26
    comp_p1=np.array([pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,SS_NLP.T[1]] + SS_NLP.P_ini[n] - SS_NLP.p_DA[n,SS_NLP.T[1]] )  for n in SS_NLP.N ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R[n,SS_NLP.T[1]] )  for n in SS_NLP.N ])
    M1_comp26=M_tune*max(comp_p1)+M_c
    M2_comp26=M_tune*max(comp_p2)+M_c 
    #comp27
    comp_p1=np.array([pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t2] + SS_NLP.p_DA[n,t1] - SS_NLP.p_DA[n,t2] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for n in SS_NLP.N ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R[n,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for n in SS_NLP.N ])
    M1_comp27=M_tune*max(comp_p1)+M_c
    M2_comp27=M_tune*max(comp_p2)+M_c
    #comp28
    comp_p1=np.array([pe.value( SS_NLP.Ramp[n]*SS_NLP.U_ini[n] + SS_NLP.p_DA[n,SS_NLP.T[1]] - SS_NLP.P_ini[n] )  for n in SS_NLP.N ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R[n,SS_NLP.T[1]] )  for n in SS_NLP.N ])
    M1_comp28=M_tune*max(comp_p1)+M_c
    M2_comp28=M_tune*max(comp_p2)+M_c
    #comp29
    comp_p1=np.array([pe.value( SS_NLP.u_DA[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_B[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    M1_comp29=M_tune*max(comp_p1)+M_c
    M2_comp29=M_tune*max(comp_p2)+M_c 
    #comp30
    comp_p1=np.array([pe.value( 1 - SS_NLP.u_DA[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_B[g,t] ) for t in SS_NLP.T for g in SS_NLP.G ])
    M1_comp30=M_tune*max(comp_p1)+M_c
    M2_comp30=M_tune*max(comp_p2)+M_c 
    
    #comp31
    comp_p1=np.array([pe.value( SS_NLP.q_DA[s,t]  ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_H[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp31=M_tune*max(comp_p1)+M_c
    M2_comp31=M_tune*max(comp_p2)+M_c 
    #comp32
    comp_p1=np.array([pe.value( SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_H[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp32=M_tune*max(comp_p1)+M_c
    M2_comp32=M_tune*max(comp_p2)+M_c
    #comp33
    comp_p1=np.array([pe.value( SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp33=M_tune*max(comp_p1)+M_c
    M2_comp33=M_tune*max(comp_p2)+M_c
    #comp34
    comp_p1=np.array([pe.value( SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp34=M_tune*max(comp_p1)+M_c
    M2_comp34=M_tune*max(comp_p2)+M_c
    #comp35
    comp_p1=np.array([pe.value( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_C[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp35=M_tune*max(comp_p1)+M_c
    M2_comp35=M_tune*max(comp_p2)+M_c
    #comp36
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    M1_comp36=M_tune*max(comp_p1)+M_c
    M2_comp36=M_tune*max(comp_p2)+M_c
    #comp37
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) )  for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_SU[s,SS_NLP.T[1]] )  for s in SS_NLP.SSC ])
    M1_comp37=M_tune*max(comp_p1)+M_c
    M2_comp37=M_tune*max(comp_p2)+M_c
    #comp38
    comp_p1=np.array([pe.value( SS_NLP.c_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_SU[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp38=M_tune*max(comp_p1)+M_c
    M2_comp38=M_tune*max(comp_p2)+M_c
    #comp39
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA[s,t2] - SS_NLP.p_DA[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    M1_comp39=M_tune*max(comp_p1)+M_c
    M2_comp39=M_tune*max(comp_p2)+M_c
    #comp40
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA[s,SS_NLP.T[1]] ) for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_R[s,SS_NLP.T[1]] ) for s in SS_NLP.SSC ])
    M1_comp40=M_tune*max(comp_p1)+M_c
    M2_comp40=M_tune*max(comp_p2)+M_c
    #comp41
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA[s,t1] - SS_NLP.p_DA[s,t2] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R[s,t1] ) for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T) for s in SS_NLP.SSC ])
    M1_comp41=M_tune*max(comp_p1)+M_c
    M2_comp41=M_tune*max(comp_p2)+M_c
    #comp42
    comp_p1=np.array([pe.value( SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] ) for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_R[s,SS_NLP.T[1]] ) for s in SS_NLP.SSC ])
    M1_comp42=M_tune*max(comp_p1)+M_c
    M2_comp42=M_tune*max(comp_p2)+M_c
    #comp43
    comp_p1=np.array([pe.value( SS_NLP.u_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_min_B[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp43=M_tune*max(comp_p1)+M_c
    M2_comp43=M_tune*max(comp_p2)+M_c
    #comp44
    comp_p1=np.array([pe.value( 1 - SS_NLP.u_DA[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    comp_p2=np.array([pe.value( SS_NLP.mu_max_B[s,t] ) for t in SS_NLP.T for s in SS_NLP.SSC ])
    M1_comp44=M_tune*max(comp_p1)+M_c
    M2_comp44=M_tune*max(comp_p2)+M_c  
    
    ##Warming: Set intial values of binary variables 

    ##Complementarity Constraints
    U1={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value( SS_NLP.mu_min_H[s,t]) > 0.001:
                U1[s,t]=0
            if pe.value( SS_NLP.q_DA[s,t]) > 0.001:
                U1[s,t]=1
    U2={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_H[s,t] ) > 0.001:
                U2[s,t]=0
            if pe.value(SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t] ) > 0.001:
                U2[s,t]=1
    U3={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_C_H[s,t] ) > 0.001:
                U3[s,t]=0
            if pe.value(SS_NLP.p_DA_H[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) > 0.001:
                U3[s,t]=1
    U4={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_min_C_H[s,t] ) > 0.001:
                U4[s,t]=0
            if pe.value(SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) > 0.001:
                U4[s,t]=1
    
    U5={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_C_H[s,t] ) > 0.001:
                U5[s,t]=0
            if pe.value(SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA_H[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] ) > 0.001:
                U5[s,t]=1
    U6={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_SU[s,t1] ) > 0.001:
                U6[s,t1]=0
            if pe.value(SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) ) > 0.001:
                U6[s,t1]=1
    U7={}
    for s in SS_NLP.NSS:
        if pe.value(SS_NLP.mu_max_SU[s,SS_NLP.T[1]] ) > 0.001:
            U7[s,SS_NLP.T[1]]=0
        if pe.value(SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) ) > 0.001:
            U7[s,SS_NLP.T[1]]=1
    U8={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_min_SU[s,t] ) > 0.001:
                U8[s,t]=0
            if pe.value(SS_NLP.c_DA[s,t] ) > 0.001:
                U8[s,t]=1
    U9={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_R_H[s,t1] ) > 0.001:
                U9[s,t1]=0
            if pe.value(SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA_H[s,t2] - SS_NLP.p_DA_H[s,t1] ) > 0.001:
                U9[s,t1]=1
    U10={}
    for s in SS_NLP.NSS:
        if pe.value( SS_NLP.mu_max_R_H[s,SS_NLP.T[1]] ) > 0.001:
            U10[s,SS_NLP.T[1]]=0
        if pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA_H[s,SS_NLP.T[1]]) > 0.001:
            U10[s,SS_NLP.T[1]]=1
    U11={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.NSS:
            if pe.value( SS_NLP.mu_min_R_H[s,t1] ) > 0.001:
                U11[s,t1]=0
            if pe.value(SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA_H[s,t1] - SS_NLP.p_DA_H[s,t2] ) > 0.001:
                U11[s,t1]=1
    U12={}
    for s in SS_NLP.NSS:
        if pe.value(SS_NLP.mu_min_R_H[s,SS_NLP.T[1]] ) > 0.001:
            U12[s,SS_NLP.T[1]]=0
        if pe.value(SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA_H[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] ) > 0.001:
            U12[s,SS_NLP.T[1]]=1
    U13={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_min_B[s,t] ) > 0.001:
                U13[s,t]=0
            if pe.value(SS_NLP.u_DA[s,t] ) > 0.001:
                U13[s,t]=1
    U14={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_B[s,t]  ) > 0.001:
                U14[s,t]=0
            if pe.value(1 - SS_NLP.u_DA[s,t]  ) > 0.001:
                U14[s,t]=1
    
    U15={}
    for t in SS_NLP.T:
        for w in SS_NLP.W:
            if pe.value(SS_NLP.mu_min_W[w,t] ) > 0.001:
                U15[w,t]=0
            if pe.value(SS_NLP.w_DA[w,t] ) > 0.001:
                U15[w,t]=1
    U16={}
    for t in SS_NLP.T:
        for w in SS_NLP.W:
            if pe.value(SS_NLP.mu_max_W[w,t] ) > 0.001:
                U16[w,t]=0
            if pe.value(SS_NLP.Wind_DA[w,t] - SS_NLP.w_DA[w,t] ) > 0.001:
                U16[w,t]=1
    U17={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            if pe.value(SS_NLP.mu_min_G[g,t] ) > 0.001:
                U17[g,t]=0
            if pe.value(SS_NLP.p_DA[g,t] - SS_NLP.P_min[g]*SS_NLP.u_DA[g,t] ) > 0.001:
                U17[g,t]=1
    U18={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            if pe.value(SS_NLP.mu_max_G[g,t] ) > 0.001:
                U18[g,t]=0
            if pe.value(SS_NLP.P_max[g]*SS_NLP.u_DA[g,t] - SS_NLP.p_DA[g,t] ) > 0.001:
                U18[g,t]=1
    U19={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_C[s,t] ) > 0.001:
                U19[s,t]=0
            if pe.value(SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) > 0.001:
                U19[s,t]=1
    U20={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_min_C[s,t]  ) > 0.001:
                U20[s,t]=0
            if pe.value(SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) > 0.001:
                U20[s,t]=1
    U21={}
    for t in SS_NLP.T:
        for s in SS_NLP.NSS:
            if pe.value(SS_NLP.mu_max_C[s,t] ) > 0.001:
                U21[s,t]=0
            if pe.value(SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] - SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] ) > 0.001:
                U21[s,t]=1
    U22={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for g in SS_NLP.G:
            if pe.value( SS_NLP.mu_max_SU[g,t1]  ) > 0.001:
                U22[g,t1]=0
            if pe.value( SS_NLP.c_DA[g,t1] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,t1] - SS_NLP.u_DA[g,t2])) > 0.001:
                U22[g,t1]=1
    U23={}
    for g in SS_NLP.G:
        if pe.value(SS_NLP.mu_max_SU[g,SS_NLP.T[1]] ) > 0.001:
            U23[g,SS_NLP.T[1]]=0
        if pe.value(SS_NLP.c_DA[g,SS_NLP.T[1]] - SS_NLP.C_SU[g]*(SS_NLP.u_DA[g,SS_NLP.T[1]] - SS_NLP.U_ini[g]) ) > 0.01:
            U23[g,SS_NLP.T[1]]=1
    U24={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            if pe.value( SS_NLP.mu_min_SU[g,t]) > 0.001:
                U24[g,t]=0
            if pe.value( SS_NLP.c_DA[g,t]) > 0.001:
                U24[g,t]=1
    U25={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for n in SS_NLP.N:
            if pe.value( SS_NLP.mu_max_R[n,t1] ) > 0.001:
                U25[n,t1]=0
            if pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t1] + SS_NLP.p_DA[n,t2] - SS_NLP.p_DA[n,t1] ) > 0.01:
                U25[n,t1]=1
    U26={}
    for n in SS_NLP.N:
        if pe.value( SS_NLP.mu_max_R[n,SS_NLP.T[1]] ) > 0.001:
            U26[n,SS_NLP.T[1]]=0
        if pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,SS_NLP.T[1]] + SS_NLP.P_ini[n] - SS_NLP.p_DA[n,SS_NLP.T[1]]) > 0.01:
            U26[n,SS_NLP.T[1]]=1
    U27={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for n in SS_NLP.N:
            if pe.value( SS_NLP.mu_min_R[n,t1] ) > 0.001:
                U27[n,t1]=0
            if pe.value( SS_NLP.Ramp[n]*SS_NLP.u_DA[n,t2] + SS_NLP.p_DA[n,t1] - SS_NLP.p_DA[n,t2]) > 0.001:
                U27[n,t1]=1
    U28={}
    for n in SS_NLP.N:
        if pe.value( SS_NLP.mu_min_R[n,SS_NLP.T[1]] ) > 0.001:
            U28[n,SS_NLP.T[1]]=0
        if pe.value( SS_NLP.Ramp[n]*SS_NLP.U_ini[n] + SS_NLP.p_DA[n,SS_NLP.T[1]] - SS_NLP.P_ini[n] ) > 0.001:
            U28[n,SS_NLP.T[1]]=1
    
    U29={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            if pe.value( SS_NLP.mu_min_B[g,t] ) > 0.001:
                U29[g,t]=0
            if pe.value( SS_NLP.u_DA[g,t] ) > 0.001:
                U29[g,t]=1  
    U30={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            if pe.value( SS_NLP.mu_max_B[g,t] ) > 0.001:
                U30[g,t]=0
            if pe.value( 1 - SS_NLP.u_DA[g,t]  ) > 0.001:
                U30[g,t]=1    
    U31={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_min_H[s,t]  ) > 0.001:
                U31[s,t]=0
            if pe.value( SS_NLP.q_DA[s,t] ) > 0.001:
                U31[s,t]=1 
    U32={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_max_H[s,t] ) > 0.001:
                U32[s,t]=0
            if pe.value( SS_NLP.Q_max[s] - SS_NLP.q_DA[s,t] ) > 0.001:
                U32[s,t]=1 
    U33={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_C[s,t] ) > 0.001:
                U33[s,t]=0
            if pe.value( SS_NLP.p_DA[s,t]-SS_NLP.r_chp[s]*SS_NLP.q_DA[s,t] ) > 0.001:
                U33[s,t]=1 
    U34={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_min_C[s,t] ) > 0.001:
                U34[s,t]=0
            if pe.value( SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] + SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t] - SS_NLP.u_DA[s,t]*SS_NLP.Fuel_min[s] ) > 0.001:
                U34[s,t]=1 
    U35={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_max_C[s,t] ) > 0.001:
                U35[s,t]=0
            if pe.value( SS_NLP.u_DA[s,t]*SS_NLP.Fuel_max[s] -  SS_NLP.rho_elec[s]*SS_NLP.p_DA[s,t] -  SS_NLP.rho_heat[s]*SS_NLP.q_DA[s,t]  ) > 0.001:
                U35[s,t]=1 
    U36={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_max_SU[s,t1] ) > 0.001:
                U36[s,t1]=0
            if pe.value( SS_NLP.c_DA[s,t1] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,t1] - SS_NLP.u_DA[s,t2]) ) > 0.001:
                U36[s,t1]=1 
    U37={}
    for s in SS_NLP.SSC:
        if pe.value( SS_NLP.mu_max_SU[s,SS_NLP.T[1]] ) > 0.001:
            U37[s,t]=0
        if pe.value( SS_NLP.c_DA[s,SS_NLP.T[1]] - SS_NLP.C_SU[s]*(SS_NLP.u_DA[s,SS_NLP.T[1]] - SS_NLP.U_ini[s]) ) > 0.001:
            U37[s,t]=1 
    U38={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_min_SU[s,t] ) > 0.001:
                U38[s,t]=0
            if pe.value( SS_NLP.c_DA[s,t] ) > 0.001:
                U38[s,t]=1 
    U39={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_max_R[s,t1] ) > 0.001:
                U39[s,t1]=0
            if pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t1] + SS_NLP.p_DA[s,t2] - SS_NLP.p_DA[s,t1] ) > 0.001:
                U39[s,t1]=1 
    U40={}
    for s in SS_NLP.SSC:
        if pe.value( SS_NLP.mu_max_R[s,SS_NLP.T[1]]  ) > 0.001:
            U40[s,SS_NLP.T[1]]=0
        if pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,SS_NLP.T[1]] + SS_NLP.P_ini[s] - SS_NLP.p_DA[s,SS_NLP.T[1]] ) > 0.001:
            U40[s,SS_NLP.T[1]]=1 
    U41={}
    for t1, t2 in zip(SS_NLP.Tnot1, SS_NLP.T):
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_min_R[s,t1] ) > 0.001:
                U41[s,t1]=0
            if pe.value( SS_NLP.Ramp[s]*SS_NLP.u_DA[s,t2] + SS_NLP.p_DA[s,t1] - SS_NLP.p_DA[s,t2] ) > 0.001:
                U41[s,t1]=1 
    U42={}
    for s in SS_NLP.SSC:
        if pe.value( SS_NLP.mu_min_R[s,SS_NLP.T[1]] ) > 0.001:
            U42[s,SS_NLP.T[1]]=0
        if pe.value( SS_NLP.Ramp[s]*SS_NLP.U_ini[s] + SS_NLP.p_DA[s,SS_NLP.T[1]] - SS_NLP.P_ini[s] ) > 0.001:
            U42[s,SS_NLP.T[1]]=1 
    U43={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_min_B[s,t] ) > 0.001:
                U43[s,t]=0
            if pe.value( SS_NLP.u_DA[s,t] ) > 0.001:
                U43[s,t]=1 
    U44={}
    for t in SS_NLP.T:
        for s in SS_NLP.SSC:
            if pe.value( SS_NLP.mu_max_B[s,t] ) > 0.001:
                U44[s,t]=0
            if pe.value( 1 - SS_NLP.u_DA[s,t] ) > 0.001:
                U44[s,t]=1 
                            
# =============================================================================
#     for t in SS_NLP.T:
#          for s in SS_NLP.CHP:
#              print('q_DA[',t,',',s,',]', pe.value(SS_NLP.q_DA[s,t]))                            
# =============================================================================
    
    total_cost_NLP=sum( SS_NLP.C_fuel[s]*(SS_NLP.rho_heat[s]*pe.value(SS_NLP.q_DA[s,t]) + SS_NLP.rho_elec[s]*pe.value(SS_NLP.p_DA[s,t])) + pe.value(SS_NLP.c_DA[s,t]) for t in SS_NLP.T for s in SS_NLP.CHP ) + sum(SS_NLP.C_fuel[g]*pe.value(SS_NLP.p_DA[g,t]) + pe.value(SS_NLP.c_DA[g,t]) for t in SS_NLP.T for g in SS_NLP.G)
    print("Total_cost_NLP = ", total_cost_NLP)
    
    lambda_DA_H_nlp={}        
    #print ("Heat_prices _NLP")
    for t in  SS_NLP.T:    
            lambda_DA_H_nlp[t] = pe.value(SS_NLP.lambda_DA_H[t])
    #print(lambda_DA_H_nlp)
        
    
    lambda_DA_E_nlp={}        
    #print ("Electricity prices _NLP")
    for t in  SS_NLP.T:    
            lambda_DA_E_nlp[t] = pe.value(SS_NLP.lambda_DA_E[t])
            
    #print(lambda_DA_E_nlp)
    
    ##Printing the variables results
    # =============================================================================
    q_DA_nlp={}
    mu_min_H_nlp={}
    mu_max_H_nlp={}
    mu_C_nlp={}
    mu_min_C_nlp={}
    mu_max_C_nlp={}
    for t in SS_NLP.T:
         for s in SS_NLP.CHP:
             q_DA_nlp[s,t] = pe.value(SS_NLP.q_DA[s,t])
             mu_min_H_nlp[s,t]=pe.value(SS_NLP.mu_min_H[s,t])
             mu_max_H_nlp[s,t]=pe.value(SS_NLP.mu_max_H[s,t])
             mu_C_nlp[s,t]=pe.value(SS_NLP.mu_C[s,t])
             mu_min_C_nlp[s,t]=pe.value(SS_NLP.mu_min_C[s,t])
             mu_max_C_nlp[s,t]=pe.value(SS_NLP.mu_max_C[s,t])
    p_DA_H_nlp={}
    mu_C_H_nlp={}
    mu_min_C_H_nlp={}
    mu_max_C_H_nlp={}
    mu_min_R_H_nlp={}
    mu_max_R_H_nlp={}
    for t in SS_NLP.T:
         for s in SS_NLP.NSS:
             p_DA_H_nlp[s,t] = pe.value(SS_NLP.p_DA_H[s,t])
             mu_C_H_nlp[s,t]=pe.value(SS_NLP.mu_C_H[s,t])
             mu_min_C_H_nlp[s,t]=pe.value(SS_NLP.mu_min_C_H[s,t])
             mu_max_C_H_nlp[s,t]=pe.value(SS_NLP.mu_max_C_H[s,t])
             mu_min_R_H_nlp[s,t]=pe.value(SS_NLP.mu_min_R_H[s,t])
             mu_max_R_H_nlp[s,t]=pe.value(SS_NLP.mu_max_R_H[s,t])
    p_DA_nlp={}
    u_DA_nlp={}
    c_DA_nlp={}
    mu_min_SU_nlp={}
    mu_max_SU_nlp={}
    mu_min_B_nlp={}
    mu_max_B_nlp={}
    mu_min_R_nlp={}
    mu_max_R_nlp={}
    for t in SS_NLP.T:
         for i in SS_NLP.I:
              p_DA_nlp[i,t] = pe.value(SS_NLP.p_DA[i,t])
              u_DA_nlp[i,t]=pe.value(SS_NLP.u_DA[i,t])
              c_DA_nlp[i,t]=pe.value(SS_NLP.c_DA[i,t])
              mu_min_SU_nlp[i,t]=pe.value(SS_NLP.mu_min_SU[i,t])
              mu_max_SU_nlp[i,t]=pe.value(SS_NLP.mu_max_SU[i,t])
              mu_min_B_nlp[i,t]=pe.value(SS_NLP.mu_min_B[i,t])
              mu_max_B_nlp[i,t]=pe.value(SS_NLP.mu_max_B[i,t])
              mu_min_R_nlp[i,t]=pe.value(SS_NLP.mu_min_R[i,t])
              mu_max_R_nlp[i,t]=pe.value(SS_NLP.mu_max_R[i,t])
    w_DA_nlp={}
    mu_min_W_nlp={}
    mu_max_W_nlp={}
    for t in SS_NLP.T:
         for w in SS_NLP.W:
              w_DA_nlp[w,t]=pe.value(SS_NLP.w_DA[w,t])
              mu_min_W_nlp[w,t]=pe.value(SS_NLP.mu_min_W[w,t])
              mu_max_W_nlp[w,t]=pe.value(SS_NLP.mu_max_W[w,t])
    mu_min_G_nlp={}
    mu_max_G_nlp={}
    for t in SS_NLP.T:
        for g in SS_NLP.G:
            mu_min_G_nlp[g,t]=pe.value(SS_NLP.mu_min_G[g,t])
            mu_max_G_nlp[g,t]=pe.value(SS_NLP.mu_max_G[g,t])
            
    #Mixed integer reformulation MODEL
    SS=pe.ConcreteModel()
    #Duals are desired
    #SS.dual = pe.Suffix(direction=pe.Suffix.IMPORT) 
    
    
    ### SETS
    SS.CHP = pe.Set(initialize = CHP) 
    SS.T = pe.Set(initialize = time)
    SS.Tnot1=pe.Set(initialize = time[1:])
    SS.T1=pe.Set(initialize = time[0:1])
    SS.I =pe.Set(initialize = I)
    
    SS.G = pe.Set(initialize = gen ) 
    SS.W = pe.Set(initialize = wind)
    
    SS.SSC = pe.Set(initialize = SSC ) 
    SS.NSS = pe.Set(initialize = NSS )
    
    SS.N = pe.Set(initialize = N )
    
    
    ### PARAMETERS
    SS.Q_max = pe.Param(SS.CHP, initialize = heat_maxprod) #Max SS production
    
    SS.Ramp = pe.Param(SS.I, initialize = Ramp)
    
    SS.Fuel_min = pe.Param(SS.CHP, initialize = Fuel_min)
    SS.Fuel_max = pe.Param(SS.CHP, initialize = Fuel_max)
    SS.rho_elec = pe.Param(SS.CHP, initialize = rho_elec) # efficiency of the CHP for SStricity production
    SS.rho_heat = pe.Param(SS.CHP, initialize = rho_heat) # efficiency of the CHP for SS production
    SS.r_chp = pe.Param(SS.CHP, initialize = r_chp) # el/heat ratio (flexible in the case of extraction units)
    
    SS.C_fuel = pe.Param(SS.I, initialize = C_fuel)
    SS.C_SU = pe.Param(SS.I, initialize = C_SU)
    SS.P_ini = pe.Param(SS.I, initialize = P_ini)
    SS.U_ini = pe.Param(SS.I, initialize = U_ini)
    
    
    
    SS.P_max = pe.Param(SS.G, initialize = elec_maxprod) #Only for Generators
    SS.P_min = pe.Param(SS.G, initialize = elec_minprod) #Only for Generators
    
    SS.D_H = pe.Param(SS.T, initialize = D_H) #SS Demand
    SS.D_E = pe.Param(SS.T, initialize = D_E) #El demand
    
    SS.Wind_DA = pe.Param(SS.W, SS.T, initialize = Wind_DA)
    
    SS.L_DA_E_F = pe.Param(SS.T, initialize = L_DA_E_F) #Electricity price Day-Ahead forecast
    
    #Big_Ms from NLP solution
    
    SS.M1_comp1 = pe.Param( initialize = M1_comp1)
    SS.M2_comp1 = pe.Param( initialize = M2_comp1)
    
    SS.M1_comp2 = pe.Param( initialize = M1_comp2)
    SS.M2_comp2 = pe.Param( initialize = M2_comp2)
    
    SS.M1_comp3 = pe.Param( initialize = M1_comp3)
    SS.M2_comp3 = pe.Param( initialize = M2_comp3)
    
    SS.M1_comp4 = pe.Param( initialize = M1_comp4)
    SS.M2_comp4 = pe.Param( initialize = M2_comp4)
    
    SS.M1_comp5 = pe.Param( initialize = M1_comp5)
    SS.M2_comp5 = pe.Param( initialize = M2_comp5)
    
    SS.M1_comp6 = pe.Param( initialize = M1_comp6)
    SS.M2_comp6 = pe.Param( initialize = M2_comp6)
    
    SS.M1_comp7 = pe.Param( initialize = M1_comp7)
    SS.M2_comp7 = pe.Param( initialize = M2_comp7)
    
    SS.M1_comp8 = pe.Param( initialize = M1_comp8)
    SS.M2_comp8 = pe.Param( initialize = M2_comp8)
    
    SS.M1_comp9 = pe.Param( initialize = M1_comp9)
    SS.M2_comp9 = pe.Param( initialize = M2_comp9)
    
    SS.M1_comp10 = pe.Param( initialize = M1_comp10)
    SS.M2_comp10 = pe.Param( initialize = M2_comp10)
    
    SS.M1_comp11 = pe.Param( initialize = M1_comp11)
    SS.M2_comp11 = pe.Param( initialize = M2_comp11)
    
    SS.M1_comp12 = pe.Param( initialize = M1_comp12)
    SS.M2_comp12 = pe.Param( initialize = M2_comp12)
    
    SS.M1_comp13 = pe.Param( initialize = M1_comp13)
    SS.M2_comp13 = pe.Param( initialize = M2_comp13)
    
    SS.M1_comp14 = pe.Param( initialize = M1_comp14)
    SS.M2_comp14 = pe.Param( initialize = M2_comp14)
    
    SS.M1_comp15 = pe.Param( initialize = M1_comp15)
    SS.M2_comp15 = pe.Param( initialize = M2_comp15)
    
    SS.M1_comp16 = pe.Param( initialize = M1_comp16)
    SS.M2_comp16 = pe.Param( initialize = M2_comp16)
    
    SS.M1_comp17 = pe.Param( initialize = M1_comp17)
    SS.M2_comp17 = pe.Param( initialize = M2_comp17)
    
    SS.M1_comp18 = pe.Param( initialize = M1_comp18)
    SS.M2_comp18 = pe.Param( initialize = M2_comp18)
    
    SS.M1_comp19 = pe.Param( initialize = M1_comp19)
    SS.M2_comp19 = pe.Param( initialize = M2_comp19)
    
    SS.M1_comp20 = pe.Param( initialize = M1_comp20)
    SS.M2_comp20 = pe.Param( initialize = M2_comp20)
    
    SS.M1_comp21 = pe.Param( initialize = M1_comp21)
    SS.M2_comp21 = pe.Param( initialize = M2_comp21)
    
    SS.M1_comp22 = pe.Param( initialize = M1_comp22)
    SS.M2_comp22 = pe.Param( initialize = M2_comp22)
    
    SS.M1_comp23 = pe.Param( initialize = M1_comp23)
    SS.M2_comp23 = pe.Param( initialize = M2_comp23)
    
    SS.M1_comp24 = pe.Param( initialize = M1_comp24)
    SS.M2_comp24 = pe.Param( initialize = M2_comp24)
    
    SS.M1_comp25 = pe.Param( initialize = M1_comp25)
    SS.M2_comp25 = pe.Param( initialize = M2_comp25)
    
    SS.M1_comp26 = pe.Param( initialize = M1_comp26)
    SS.M2_comp26 = pe.Param( initialize = M2_comp26)
    
    SS.M1_comp27 = pe.Param( initialize = M1_comp27)
    SS.M2_comp27 = pe.Param( initialize = M2_comp27)
    
    SS.M1_comp28 = pe.Param( initialize = M1_comp28)
    SS.M2_comp28 = pe.Param( initialize = M2_comp28)
    
    SS.M1_comp29 = pe.Param( initialize = M1_comp29)
    SS.M2_comp29 = pe.Param( initialize = M2_comp29)
    
    SS.M1_comp30 = pe.Param( initialize = M1_comp30)
    SS.M2_comp30 = pe.Param( initialize = M2_comp30)
    
    SS.M1_comp31 = pe.Param( initialize = M1_comp31)
    SS.M2_comp31 = pe.Param( initialize = M2_comp31)
    
    SS.M1_comp32 = pe.Param( initialize = M1_comp32)
    SS.M2_comp32 = pe.Param( initialize = M2_comp32)
    
    SS.M1_comp33 = pe.Param( initialize = M1_comp33)
    SS.M2_comp33 = pe.Param( initialize = M2_comp33)
    
    SS.M1_comp34 = pe.Param( initialize = M1_comp34)
    SS.M2_comp34 = pe.Param( initialize = M2_comp34)
    
    SS.M1_comp35 = pe.Param( initialize = M1_comp35)
    SS.M2_comp35 = pe.Param( initialize = M2_comp35)
    
    SS.M1_comp36 = pe.Param( initialize = M1_comp36)
    SS.M2_comp36 = pe.Param( initialize = M2_comp36)
    
    SS.M1_comp37 = pe.Param( initialize = M1_comp37)
    SS.M2_comp37 = pe.Param( initialize = M2_comp37)
    
    SS.M1_comp38 = pe.Param( initialize = M1_comp38)
    SS.M2_comp38 = pe.Param( initialize = M2_comp38)
    
    SS.M1_comp39 = pe.Param( initialize = M1_comp39)
    SS.M2_comp39 = pe.Param( initialize = M2_comp39)
    
    SS.M1_comp40 = pe.Param( initialize = M1_comp40)
    SS.M2_comp40 = pe.Param( initialize = M2_comp40)
    
    SS.M1_comp41 = pe.Param( initialize = M1_comp41)
    SS.M2_comp41 = pe.Param( initialize = M2_comp41)
    
    SS.M1_comp42 = pe.Param( initialize = M1_comp42)
    SS.M2_comp42 = pe.Param( initialize = M2_comp42)
    
    SS.M1_comp43 = pe.Param( initialize = M1_comp43)
    SS.M2_comp43 = pe.Param( initialize = M2_comp43)
    
    SS.M1_comp44 = pe.Param( initialize = M1_comp44)
    SS.M2_comp44 = pe.Param( initialize = M2_comp44)
    
    ### VARIABLES
    def _bounds_rule_q(m, chp, t):
        return (0, heat_maxprod[chp])
    SS.q_DA=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=_bounds_rule_q, initialize =  q_DA_nlp)
    
    SS.p_DA_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  p_DA_H_nlp)
    
    SS.p_DA=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  p_DA_nlp)
    def _bounds_rule_w(m, j, t):
        return (0, Wind_DA[j,t])
    ##def _init_rule_W_DA(m, j, t):
     #   return (Wind_DA[j,t])
    SS.w_DA=pe.Var(SS.W, SS.T, domain=pe.NonNegativeReals, bounds=_bounds_rule_w, initialize =  w_DA_nlp)
    
    SS.u_DA=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0,1), initialize =  u_DA_nlp)
    
    #def _bounds_rule_c(m, j, t):
    #    return (0, C_SU[j])
    SS.c_DA=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  c_DA_nlp)
    
    
    
    ##Dual variables
    #positive variables
    SS.mu_min_H=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_H_nlp)
    SS.mu_max_H=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_H_nlp)
    
    SS.mu_C_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_C_H_nlp)
    
    SS.mu_min_C_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_C_H_nlp)
    SS.mu_max_C_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_C_H_nlp)
    
    SS.mu_min_R_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_R_H_nlp)
    SS.mu_max_R_H=pe.Var(SS.NSS, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_R_H_nlp)
    
    SS.mu_C=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_C_nlp)
    
    SS.mu_min_C=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_C_nlp)
    SS.mu_max_C=pe.Var(SS.CHP, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_C_nlp)
    
    SS.mu_min_W=pe.Var(SS.W, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_W_nlp)
    
    SS.mu_max_W=pe.Var(SS.W, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_W_nlp)
    
    
    SS.mu_min_G=pe.Var(SS.G, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_G_nlp)
    SS.mu_max_G=pe.Var(SS.G, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_G_nlp)
    
    SS.mu_min_SU=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_SU_nlp)
    SS.mu_max_SU=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_SU_nlp)
    
    SS.mu_min_B=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_B_nlp)
    SS.mu_max_B=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_B_nlp)
    
    SS.mu_min_R=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_min_R_nlp)
    SS.mu_max_R=pe.Var(SS.I, SS.T, domain=pe.NonNegativeReals, bounds=(0, None), initialize =  mu_max_R_nlp)
    
    
    
    #Integer auxiliary variables 
    SS.u_comp1=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U1)
    SS.u_comp2=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U2)
    SS.u_comp3=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U3)
    SS.u_comp4=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U4)
    SS.u_comp5=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U5)
    SS.u_comp6=pe.Var(SS.NSS, SS.Tnot1, domain=pe.Binary, initialize= U6)
    
    SS.u_comp7=pe.Var(SS.NSS, SS.T1, domain=pe.Binary, initialize= U7)
    
    SS.u_comp8=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U8)
    SS.u_comp9=pe.Var(SS.NSS, SS.Tnot1, domain=pe.Binary, initialize= U9)
    
    SS.u_comp10=pe.Var(SS.NSS, SS.T1, domain=pe.Binary, initialize= U10)
    
    SS.u_comp11=pe.Var(SS.NSS, SS.Tnot1, domain=pe.Binary, initialize= U11)
    
    SS.u_comp12=pe.Var(SS.NSS, SS.T1, domain=pe.Binary, initialize= U12)
    
    SS.u_comp13=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U13)
    SS.u_comp14=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U14)
    
    SS.u_comp15=pe.Var(SS.W, SS.T, domain=pe.Binary, initialize= U15)
    SS.u_comp16=pe.Var(SS.W, SS.T, domain=pe.Binary, initialize= U16)
    
    SS.u_comp17=pe.Var(SS.G, SS.T, domain=pe.Binary, initialize= U17)
    SS.u_comp18=pe.Var(SS.G, SS.T, domain=pe.Binary, initialize= U18)
    
    SS.u_comp19=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U19)
    SS.u_comp20=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U20)
    SS.u_comp21=pe.Var(SS.NSS, SS.T, domain=pe.Binary, initialize= U21)
    
    SS.u_comp22=pe.Var(SS.G, SS.Tnot1, domain=pe.Binary, initialize= U22)
    
    SS.u_comp23=pe.Var(SS.G, SS.T1, domain=pe.Binary, initialize= U23)
    
    SS.u_comp24=pe.Var(SS.G, SS.T, domain=pe.Binary, initialize= U24)
    
    SS.u_comp25=pe.Var(SS.N, SS.Tnot1, domain=pe.Binary, initialize= U25)
    
    SS.u_comp26=pe.Var(SS.N, SS.T1, domain=pe.Binary, initialize= U26)
    
    SS.u_comp27=pe.Var(SS.N, SS.Tnot1, domain=pe.Binary, initialize= U27)
    
    SS.u_comp28=pe.Var(SS.N, SS.T1, domain=pe.Binary, initialize= U28)
    
    SS.u_comp29=pe.Var(SS.G, SS.T, domain=pe.Binary, initialize= U29)
    SS.u_comp30=pe.Var(SS.G, SS.T, domain=pe.Binary, initialize= U30)
    
    SS.u_comp31=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U31)
    SS.u_comp32=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U32)
    SS.u_comp33=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U33)
    SS.u_comp34=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U34)
    SS.u_comp35=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U35)
    SS.u_comp36=pe.Var(SS.SSC, SS.Tnot1, domain=pe.Binary, initialize= U36)
    
    SS.u_comp37=pe.Var(SS.SSC, SS.T1, domain=pe.Binary, initialize= U37)
    
    SS.u_comp38=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U38)
    SS.u_comp39=pe.Var(SS.SSC, SS.Tnot1, domain=pe.Binary, initialize= U39)
    
    SS.u_comp40=pe.Var(SS.SSC, SS.T1, domain=pe.Binary, initialize= U40)
    
    SS.u_comp41=pe.Var(SS.SSC, SS.Tnot1, domain=pe.Binary, initialize= U41)
    
    SS.u_comp42=pe.Var(SS.SSC, SS.T1, domain=pe.Binary, initialize= U42)
    
    SS.u_comp43=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U43)
    SS.u_comp44=pe.Var(SS.SSC, SS.T, domain=pe.Binary, initialize= U44)
    
    #free variables
    def _init_rule_L_E(m, t):
        return (lambda_DA_E_nlp[t])
    SS.lambda_DA_E=pe.Var(SS.T, domain=pe.Reals,  initialize=_init_rule_L_E) # day-ahead electricity price in period t [$ per MWh]
    
    def _init_rule_L_H(m, t):
        return (lambda_DA_H_nlp[t])
    SS.lambda_DA_H=pe.Var(SS.T, domain=pe.Reals, initialize=_init_rule_L_H) #day-ahead heat price in period t [$ per MWh]
    
    ##CONSTRAINTS
    
    ##KKT HEAT MARKET
    ##Stationarity conditions
            
    SS.L1_q_DA = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
            SS.L1_q_DA.add( SS.C_fuel[s]*SS.rho_heat[s] - SS.mu_min_H[s,t] + SS.mu_max_H[s,t] - SS.lambda_DA_H[t] + SS.mu_C_H[s,t]*SS.r_chp[s] - SS.mu_min_C_H[s,t]*SS.rho_heat[s] + SS.mu_max_C_H[s,t]*SS.rho_heat[s] == 0)       
    
    SS.L1_p_DA = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for s in SS.NSS:
            SS.L1_p_DA.add( SS.C_fuel[s]*SS.rho_elec[s] - SS.L_DA_E_F[t1] - SS.mu_C_H[s,t1] - SS.mu_min_C_H[s,t1]*SS.rho_elec[s] + SS.mu_max_C_H[s,t1]*SS.rho_elec[s] \
                            + SS.mu_max_R_H[s,t1]  - SS.mu_max_R_H[s,t2] + SS.mu_min_R_H[s,t2] - SS.mu_min_R_H[s,t1] == 0 )       
    #t24
    SS.L1_p_DA1 = pe.ConstraintList()
    for s in SS.NSS:
        SS.L1_p_DA1.add( SS.C_fuel[s]*SS.rho_elec[s] - SS.L_DA_E_F[SS.T[T]] - SS.mu_C_H[s,SS.T[T]] - SS.mu_min_C_H[s,SS.T[T]]*SS.rho_elec[s] + SS.mu_max_C_H[s,SS.T[T]]*SS.rho_elec[s] \
                        + SS.mu_max_R_H[s,SS.T[T]] - SS.mu_min_R_H[s,SS.T[T]] == 0)       
    
    SS.L1_u_DA = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for s in SS.NSS:
            SS.L1_u_DA.add( SS.mu_min_C_H[s,t1]*SS.Fuel_min[s] - SS.mu_max_C_H[s,t1]*SS.Fuel_max[s] + SS.mu_max_SU[s,t1]*SS.C_SU[s] - SS.mu_max_SU[s,t2]*SS.C_SU[s] - SS.mu_min_B[s,t1] + SS.mu_max_B[s,t1] \
                           - SS.mu_max_R_H[s,t1]*SS.Ramp[s] - SS.mu_min_R_H[s,t2]*SS.Ramp[s] == 0 )
    #t24
    SS.L1_u_DA1 = pe.ConstraintList()
    for s in SS.NSS:
        SS.L1_u_DA1.add( SS.mu_min_C_H[s,SS.T[T]]*SS.Fuel_min[s] - SS.mu_max_C_H[s,SS.T[T]]*SS.Fuel_max[s] + SS.mu_max_SU[s,SS.T[T]]*SS.C_SU[s] - SS.mu_min_B[s,SS.T[T]] + SS.mu_max_B[s,SS.T[T]] \
                       - SS.mu_max_R_H[s,SS.T[T]]*SS.Ramp[s] == 0 )
            
    SS.L1_c_DA = pe.ConstraintList()   
    for t in SS.T:
        for s in SS.NSS:
            SS.L1_c_DA.add( 1 - SS.mu_max_SU[s,t] - SS.mu_min_SU[s,t] == 0 )
            
    
    ##Complementarity Constraint
    
    SS.comp1_1 = pe.ConstraintList()
    SS.comp1_2 = pe.ConstraintList()
    SS.comp1_3 = pe.ConstraintList()
    SS.comp1_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp1_1.add( SS.q_DA[s,t] >= 0)
           SS.comp1_2.add( SS.mu_min_H[s,t] >= 0)
           SS.comp1_3.add( SS.q_DA[s,t] <= SS.u_comp1[s,t]*SS.M1_comp1)
           SS.comp1_4.add( SS.mu_min_H[s,t] <= (1-SS.u_comp1[s,t])*SS.M2_comp1 )
    
    SS.comp2_1 = pe.ConstraintList()
    SS.comp2_2 = pe.ConstraintList()
    SS.comp2_3 = pe.ConstraintList()
    SS.comp2_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
            SS.comp2_1.add( SS.Q_max[s] - SS.q_DA[s,t] >= 0)
            SS.comp2_2.add( SS.mu_max_H[s,t] >= 0)
            SS.comp2_3.add( SS.Q_max[s] - SS.q_DA[s,t] <= SS.u_comp2[s,t]*SS.M1_comp2 )
            SS.comp2_4.add( SS.mu_max_H[s,t] <= (1-SS.u_comp2[s,t])*SS.M2_comp2 )
                
    SS.comp3_1 = pe.ConstraintList()
    SS.comp3_2 = pe.ConstraintList()
    SS.comp3_3 = pe.ConstraintList()
    SS.comp3_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp3_1.add( SS.p_DA_H[s,t]-SS.r_chp[s]*SS.q_DA[s,t] >= 0)
           SS.comp3_2.add( SS.mu_C_H[s,t] >= 0)
           SS.comp3_3.add( SS.p_DA_H[s,t]-SS.r_chp[s]*SS.q_DA[s,t] <= SS.u_comp3[s,t]*SS.M1_comp3 )
           SS.comp3_4.add( SS.mu_C_H[s,t] <= (1-SS.u_comp3[s,t])*SS.M2_comp3  )
            
    SS.comp4_1 = pe.ConstraintList()
    SS.comp4_2 = pe.ConstraintList()
    SS.comp4_3 = pe.ConstraintList()
    SS.comp4_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
            SS.comp4_1.add( SS.rho_elec[s]*SS.p_DA_H[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] >= 0)
            SS.comp4_2.add( SS.mu_min_C_H[s,t] >= 0)
            SS.comp4_3.add( SS.rho_elec[s]*SS.p_DA_H[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] <= SS.u_comp4[s,t]*SS.M1_comp4 )
            SS.comp4_4.add( SS.mu_min_C_H[s,t] <= (1-SS.u_comp4[s,t])*SS.M2_comp4)
    
    SS.comp5_1 = pe.ConstraintList()
    SS.comp5_2 = pe.ConstraintList()
    SS.comp5_3 = pe.ConstraintList()
    SS.comp5_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp5_1.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA_H[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t]>= 0 )
           SS.comp5_2.add( SS.mu_max_C_H[s,t] >= 0 )
           SS.comp5_3.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA_H[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t] <= SS.u_comp5[s,t]*SS.M1_comp5 )
           SS.comp5_4.add( SS.mu_max_C_H[s,t] <= (1-SS.u_comp5[s,t])*SS.M2_comp5 )
            
    
    SS.comp6_1 = pe.ConstraintList()    
    SS.comp6_2 = pe.ConstraintList() 
    SS.comp6_3 = pe.ConstraintList() 
    SS.comp6_4 = pe.ConstraintList()  
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.NSS:
            SS.comp6_1.add( SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]) >= 0)
            SS.comp6_2.add( SS.mu_max_SU[s,t1] >= 0)
            SS.comp6_3.add( SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]) <= SS.u_comp6[s,t1]*SS.M1_comp6 )
            SS.comp6_4.add( SS.mu_max_SU[s,t1] <= (1-SS.u_comp6[s,t1])*SS.M2_comp6 )
          
    SS.comp7_1 = pe.ConstraintList()     
    SS.comp7_2 = pe.ConstraintList() 
    SS.comp7_3 = pe.ConstraintList() 
    SS.comp7_4 = pe.ConstraintList() 
    
    for s in SS.NSS:
        SS.comp7_1.add( SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]) >= 0)
        SS.comp7_2.add( SS.mu_max_SU[s,SS.T[1]] >= 0)
        SS.comp7_3.add( SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]) <= SS.u_comp7[s,SS.T[1]]*SS.M1_comp7 )
        SS.comp7_4.add( SS.mu_max_SU[s,SS.T[1]] <= (1-SS.u_comp7[s,SS.T[1]])*SS.M2_comp7 )
             
    SS.comp8_1 = pe.ConstraintList()
    SS.comp8_2 = pe.ConstraintList()
    SS.comp8_3 = pe.ConstraintList()
    SS.comp8_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp8_1.add( SS.c_DA[s,t] >= 0 )
           SS.comp8_2.add( SS.mu_min_SU[s,t] >= 0 )
           SS.comp8_3.add( SS.c_DA[s,t] <= SS.u_comp8[s,t]*SS.M1_comp8 )
           SS.comp8_4.add( SS.mu_min_SU[s,t] <= (1-SS.u_comp8[s,t])*SS.M2_comp8 )
         
    SS.comp9_1 = pe.ConstraintList()    
    SS.comp9_2 = pe.ConstraintList() 
    SS.comp9_3 = pe.ConstraintList() 
    SS.comp9_4 = pe.ConstraintList()  
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.NSS:
            SS.comp9_1.add( SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA_H[s,t2] - SS.p_DA_H[s,t1] >=0)
            SS.comp9_2.add( SS.mu_max_R_H[s,t1] >= 0)
            SS.comp9_3.add( SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA_H[s,t2] - SS.p_DA_H[s,t1] <= SS.u_comp9[s,t1]*SS.M1_comp9 )
            SS.comp9_4.add( SS.mu_max_R_H[s,t1] <= (1-SS.u_comp9[s,t1])*SS.M2_comp9 )
                         
    SS.comp10_1 = pe.ConstraintList()    
    SS.comp10_2 = pe.ConstraintList() 
    SS.comp10_3 = pe.ConstraintList() 
    SS.comp10_4 = pe.ConstraintList()  
    for s in SS.NSS:
        SS.comp10_1.add( SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA_H[s,SS.T[1]] >=0 )
        SS.comp10_2.add( SS.mu_max_R_H[s,SS.T[1]] >= 0 )
        SS.comp10_3.add( SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA_H[s,SS.T[1]] <= SS.u_comp10[s,SS.T[1]]*SS.M1_comp10 )
        SS.comp10_4.add( SS.mu_max_R_H[s,SS.T[1]] <= (1-SS.u_comp10[s,SS.T[1]])*SS.M2_comp10 )
    
    SS.comp11_1 = pe.ConstraintList()
    SS.comp11_2 = pe.ConstraintList()
    SS.comp11_3 = pe.ConstraintList()
    SS.comp11_4 = pe.ConstraintList()
      
       
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.NSS:
            SS.comp11_1.add( SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA_H[s,t1] - SS.p_DA_H[s,t2] >=0)
            SS.comp11_2.add( SS.mu_min_R_H[s,t1] >=0 )
            SS.comp11_3.add( SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA_H[s,t1] - SS.p_DA_H[s,t2] <= SS.u_comp11[s,t1]*SS.M1_comp11 )
            SS.comp11_4.add( SS.mu_min_R_H[s,t1] <= (1-SS.u_comp11[s,t1])*SS.M2_comp11 )
                         
    SS.comp12_1 = pe.ConstraintList()   
    SS.comp12_2 = pe.ConstraintList() 
    SS.comp12_3 = pe.ConstraintList() 
    SS.comp12_4 = pe.ConstraintList() 
      
    for s in SS.NSS:
        SS.comp12_1.add( SS.Ramp[s]*SS.U_ini[s] + SS.p_DA_H[s,SS.T[1]] - SS.P_ini[s] >= 0 )
        SS.comp12_2.add( SS.mu_min_R_H[s,SS.T[1]] >=0 )
        SS.comp12_3.add( SS.Ramp[s]*SS.U_ini[s] + SS.p_DA_H[s,SS.T[1]] - SS.P_ini[s] <= SS.u_comp12[s,SS.T[1]]*SS.M1_comp12 )
        SS.comp12_4.add( SS.mu_min_R_H[s,SS.T[1]] <= (1-SS.u_comp12[s,SS.T[1]])*SS.M2_comp12 )
    
    SS.comp13_1 = pe.ConstraintList()
    SS.comp13_2 = pe.ConstraintList()
    SS.comp13_3 = pe.ConstraintList()
    SS.comp13_4 = pe.ConstraintList()
    
    for t in SS.T:
        for s in SS.NSS:
           SS.comp13_1.add( SS.u_DA[s,t] >= 0)
           SS.comp13_2.add( SS.mu_min_B[s,t] >= 0 )
           SS.comp13_3.add( SS.u_DA[s,t] <= SS.u_comp13[s,t]*SS.M1_comp13  )
           SS.comp13_4.add( SS.mu_min_B[s,t] <= (1-SS.u_comp13[s,t])*SS.M2_comp13 )
            
    SS.comp14_1 = pe.ConstraintList()
    SS.comp14_2 = pe.ConstraintList()
    SS.comp14_3 = pe.ConstraintList()
    SS.comp14_4 = pe.ConstraintList()
    
    for t in SS.T:
        for s in SS.NSS:
            SS.comp14_1.add( 1 - SS.u_DA[s,t] >= 0 )
            SS.comp14_2.add( SS.mu_max_B[s,t] >= 0 )
            SS.comp14_3.add( 1 - SS.u_DA[s,t] <= SS.u_comp14[s,t]*SS.M1_comp14  )
            SS.comp14_4.add( SS.mu_max_B[s,t] <= (1-SS.u_comp14[s,t])*SS.M2_comp14 )
    
    ##KKT ELECTRICITY MARKET   
    ##Stationarity conditions
                 
    SS.L2_p_DAs = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for s in SS.NSS:
            SS.L2_p_DAs.add( SS.C_fuel[s]*SS.rho_elec[s] - SS.lambda_DA_E[t1] - SS.mu_C[s,t1] - SS.mu_min_C[s,t1]*SS.rho_elec[s] + SS.mu_max_C[s,t1]*SS.rho_elec[s] \
                            + SS.mu_max_R[s,t1]  - SS.mu_max_R[s,t2] + SS.mu_min_R[s,t2] - SS.mu_min_R[s,t1] == 0 )
                
    #t24
    SS.L2_p_DAs1 = pe.ConstraintList()
    for s in SS.NSS:
        SS.L2_p_DAs1.add( SS.C_fuel[s]*SS.rho_elec[s] - SS.lambda_DA_E[SS.T[T]] - SS.mu_C[s,SS.T[T]] - SS.mu_min_C[s,SS.T[T]]*SS.rho_elec[s] + SS.mu_max_C[s,SS.T[T]]*SS.rho_elec[s] \
                        + SS.mu_max_R[s,SS.T[T]] - SS.mu_min_R[s,SS.T[T]] == 0)  
            
    
    SS.L2_p_DAg = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for g in SS.G:
            SS.L2_p_DAg.add( SS.C_fuel[g] - SS.lambda_DA_E[t1] - SS.mu_min_G[g,t1] + SS.mu_max_G[g,t1] \
                            + SS.mu_max_R[g,t1]  - SS.mu_max_R[g,t2] + SS.mu_min_R[g,t2] - SS.mu_min_R[g,t1] == 0 )    
                
    #t24
    SS.L2_p_DAg1 = pe.ConstraintList()
    for g in SS.G:
        SS.L2_p_DAg1.add( SS.C_fuel[g] - SS.lambda_DA_E[SS.T[T]]  - SS.mu_min_G[g,SS.T[T]] + SS.mu_max_G[g,SS.T[T]] \
                        + SS.mu_max_R[g,SS.T[T]] - SS.mu_min_R[g,SS.T[T]] == 0)  
                                                     
    
    SS.L2_u_DAg = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for g in SS.G:
            SS.L2_u_DAg.add( SS.mu_min_G[g,t1]*SS.P_min[g] - SS.mu_max_G[g,t1]*SS.P_max[g] + SS.mu_max_SU[g,t1]*SS.C_SU[g] - SS.mu_max_SU[g,t2]*SS.C_SU[g] - SS.mu_min_B[g,t1] + SS.mu_max_B[g,t1] \
                           - SS.mu_max_R[g,t1]*SS.Ramp[g] - SS.mu_min_R[g,t2]*SS.Ramp[g] == 0 )
                           
    #t24
    SS.L2_u_DAg1 = pe.ConstraintList()
    for g in SS.G:
        SS.L2_u_DAg1.add( SS.mu_min_G[g,SS.T[T]]*SS.P_min[g] - SS.mu_max_G[g,SS.T[T]]*SS.P_max[g] + SS.mu_max_SU[g,SS.T[T]]*SS.C_SU[g] - SS.mu_min_B[g,SS.T[T]] + SS.mu_max_B[g,SS.T[T]] \
                       - SS.mu_max_R[g,SS.T[T]]*SS.Ramp[g] == 0 )
            
            
            
    SS.L2_w_DA = pe.ConstraintList()
    for t in SS.T:
        for w in SS.W:
            SS.L2_w_DA.add( -SS.lambda_DA_E[t] - SS.mu_min_W[w,t] + SS.mu_max_W[w,t] == 0)
        
    SS.L2_c_DA = pe.ConstraintList()   
    for t in SS.T:
        for g in SS.G:
            SS.L2_c_DA.add( 1 - SS.mu_max_SU[g,t] - SS.mu_min_SU[g,t] == 0 )        
        
    ##Complementarity conditions
    SS.comp15_1 = pe.ConstraintList()
    SS.comp15_2 = pe.ConstraintList()
    SS.comp15_3 = pe.ConstraintList()
    SS.comp15_4 = pe.ConstraintList()
    for t in SS.T:
        for w in SS.W:
           SS.comp15_1.add( SS.w_DA[w,t] >= 0 )
           SS.comp15_2.add( SS.mu_min_W[w,t] >= 0 )
           SS.comp15_3.add( SS.w_DA[w,t] <= SS.u_comp15[w,t]*SS.M1_comp15  )
           SS.comp15_4.add( SS.mu_min_W[w,t] <= (1-SS.u_comp15[w,t])*SS.M2_comp15 )
            
    SS.comp16_1 = pe.ConstraintList()
    SS.comp16_2 = pe.ConstraintList()
    SS.comp16_3 = pe.ConstraintList()
    SS.comp16_4 = pe.ConstraintList()
    for t in SS.T:
        for w in SS.W:
            SS.comp16_1.add( SS.Wind_DA[w,t] - SS.w_DA[w,t] >= 0 )
            SS.comp16_2.add( SS.mu_max_W[w,t] >= 0 )
            SS.comp16_3.add( SS.Wind_DA[w,t] - SS.w_DA[w,t] <= SS.u_comp16[w,t]*SS.M1_comp16 )
            SS.comp16_4.add( SS.mu_max_W[w,t] <= (1-SS.u_comp16[w,t])*SS.M2_comp16 )
    
    SS.comp17_1 = pe.ConstraintList()
    SS.comp17_2 = pe.ConstraintList()
    SS.comp17_3 = pe.ConstraintList()
    SS.comp17_4 = pe.ConstraintList()
    for t in SS.T:
        for g in SS.G:
           SS.comp17_1.add( SS.p_DA[g,t] - SS.P_min[g]*SS.u_DA[g,t] >= 0 )
           SS.comp17_2.add( SS.mu_min_G[g,t] >= 0 )
           SS.comp17_3.add( SS.p_DA[g,t] - SS.P_min[g]*SS.u_DA[g,t] <= SS.u_comp17[g,t]*SS.M1_comp17 )
           SS.comp17_4.add( SS.mu_min_G[g,t] <= (1-SS.u_comp17[g,t])*SS.M2_comp17 )
            
    SS.comp18_1 = pe.ConstraintList()
    SS.comp18_2 = pe.ConstraintList()
    SS.comp18_3 = pe.ConstraintList()
    SS.comp18_4 = pe.ConstraintList()
    for t in SS.T:
        for g in SS.G:
            SS.comp18_1.add( SS.P_max[g]*SS.u_DA[g,t] - SS.p_DA[g,t] >= 0 )
            SS.comp18_2.add( SS.mu_max_G[g,t] >= 0 )
            SS.comp18_3.add( SS.P_max[g]*SS.u_DA[g,t] - SS.p_DA[g,t] <= SS.u_comp18[g,t]*SS.M1_comp18)
            SS.comp18_4.add( SS.mu_max_G[g,t] <= (1-SS.u_comp18[g,t])*SS.M2_comp18 )
    
    SS.comp19_1 = pe.ConstraintList()
    SS.comp19_2 = pe.ConstraintList()
    SS.comp19_3 = pe.ConstraintList()
    SS.comp19_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp19_1.add( SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t] >= 0 )
           SS.comp19_2.add( SS.mu_C[s,t] >= 0 )
           SS.comp19_3.add( SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t] <= SS.u_comp19[s,t]*SS.M1_comp19 )
           SS.comp19_4.add( SS.mu_C[s,t] <= (1-SS.u_comp19[s,t])*SS.M2_comp19 )
            
    SS.comp20_1 = pe.ConstraintList()
    SS.comp20_2 = pe.ConstraintList()
    SS.comp20_3 = pe.ConstraintList()
    SS.comp20_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
            SS.comp20_1.add( SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] >= 0 )
            SS.comp20_2.add( SS.mu_min_C[s,t] >= 0 )
            SS.comp20_3.add( SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] <= SS.u_comp20[s,t]*SS.M1_comp20 )
            SS.comp20_4.add( SS.mu_min_C[s,t] <= (1-SS.u_comp20[s,t])*SS.M2_comp20 )
    
    SS.comp21_1 = pe.ConstraintList()
    SS.comp21_2 = pe.ConstraintList()
    SS.comp21_3 = pe.ConstraintList()
    SS.comp21_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.NSS:
           SS.comp21_1.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] - SS.rho_heat[s]*SS.q_DA[s,t] >= 0)
           SS.comp21_2.add( SS.mu_max_C[s,t] >= 0 )
           SS.comp21_3.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] - SS.rho_heat[s]*SS.q_DA[s,t] <= SS.u_comp21[s,t]*SS.M1_comp21 )
           SS.comp21_4.add( SS.mu_max_C[s,t] <= (1-SS.u_comp21[s,t])*SS.M2_comp21 )
    
    SS.comp22_1 = pe.ConstraintList() 
    SS.comp22_2 = pe.ConstraintList()
    SS.comp22_3 = pe.ConstraintList()
    SS.comp22_4 = pe.ConstraintList()
        
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for g in SS.G:
            SS.comp22_1.add( SS.c_DA[g,t1] - SS.C_SU[g]*(SS.u_DA[g,t1] - SS.u_DA[g,t2]) >= 0 )
            SS.comp22_2.add( SS.mu_max_SU[g,t1] >= 0 )
            SS.comp22_3.add( SS.c_DA[g,t1] - SS.C_SU[g]*(SS.u_DA[g,t1] - SS.u_DA[g,t2]) <= SS.u_comp22[g,t1]*SS.M1_comp22 )
            SS.comp22_4.add( SS.mu_max_SU[g,t1] <= (1-SS.u_comp22[g,t1])*SS.M2_comp22 )
          
    SS.comp23_1 = pe.ConstraintList() 
    SS.comp23_2 = pe.ConstraintList() 
    SS.comp23_3 = pe.ConstraintList() 
    SS.comp23_4 = pe.ConstraintList() 
        
    for g in SS.G:
        SS.comp23_1.add( SS.c_DA[g,SS.T[1]] - SS.C_SU[g]*(SS.u_DA[g,SS.T[1]] - SS.U_ini[g]) >= 0 )
        SS.comp23_2.add( SS.mu_max_SU[g,SS.T[1]] >= 0 )
        SS.comp23_3.add( SS.c_DA[g,SS.T[1]] - SS.C_SU[g]*(SS.u_DA[g,SS.T[1]] - SS.U_ini[g]) <= SS.u_comp23[g,SS.T[1]]*SS.M1_comp23  )
        SS.comp23_4.add( SS.mu_max_SU[g,SS.T[1]] <= (1-SS.u_comp23[g,SS.T[1]])*SS.M2_comp23 )
             
    SS.comp24_1 = pe.ConstraintList()
    SS.comp24_2 = pe.ConstraintList()
    SS.comp24_3 = pe.ConstraintList()
    SS.comp24_4 = pe.ConstraintList()
    
    for t in SS.T:
        for g in SS.G:
           SS.comp24_1.add( SS.c_DA[g,t] >= 0 )
           SS.comp24_2.add( SS.mu_min_SU[g,t] >= 0 )
           SS.comp24_3.add( SS.c_DA[g,t] <= SS.u_comp24[g,t]*SS.M1_comp24 )
           SS.comp24_4.add( SS.mu_min_SU[g,t] <= (1-SS.u_comp24[g,t])*SS.M2_comp24 )
    
    
    
    SS.comp25_1 = pe.ConstraintList() 
    SS.comp25_2 = pe.ConstraintList()
    SS.comp25_3 = pe.ConstraintList()
    SS.comp25_4 = pe.ConstraintList()   
     
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for n in SS.N:
            SS.comp25_1.add( SS.Ramp[n]*SS.u_DA[n,t1] + SS.p_DA[n,t2] - SS.p_DA[n,t1] >=0 )
            SS.comp25_2.add( SS.mu_max_R[n,t1] >= 0 )
            SS.comp25_3.add( SS.Ramp[n]*SS.u_DA[n,t1] + SS.p_DA[n,t2] - SS.p_DA[n,t1] <= SS.u_comp25[n,t1]*SS.M1_comp25 )
            SS.comp25_4.add( SS.mu_max_R[n,t1] <= (1-SS.u_comp25[n,t1])*SS.M2_comp25 )
     
                        
    SS.comp26_1 = pe.ConstraintList() 
    SS.comp26_2 = pe.ConstraintList()
    SS.comp26_3 = pe.ConstraintList()
    SS.comp26_4 = pe.ConstraintList()    
    
    for n in SS.N:
        SS.comp26_1.add( SS.Ramp[n]*SS.u_DA[n,SS.T[1]] + SS.P_ini[n] - SS.p_DA[n,SS.T[1]] >=0 )
        SS.comp26_2.add( SS.mu_max_R[n,SS.T[1]] >= 0 )
        SS.comp26_3.add( SS.Ramp[n]*SS.u_DA[n,SS.T[1]] + SS.P_ini[n] - SS.p_DA[n,SS.T[1]] <= SS.u_comp26[n,SS.T[1]]*SS.M1_comp26 )
        SS.comp26_4.add( SS.mu_max_R[n,SS.T[1]] <= (1-SS.u_comp26[n,SS.T[1]])*SS.M2_comp26 )
    
    SS.comp27_1 = pe.ConstraintList() 
    SS.comp27_2 = pe.ConstraintList() 
    SS.comp27_3 = pe.ConstraintList() 
    SS.comp27_4 = pe.ConstraintList()    
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for n in SS.N:
            SS.comp27_1.add( SS.Ramp[n]*SS.u_DA[n,t2] + SS.p_DA[n,t1] - SS.p_DA[n,t2] >=0 )
            SS.comp27_2.add( SS.mu_min_R[n,t1] >=0 )
            SS.comp27_3.add( SS.Ramp[n]*SS.u_DA[n,t2] + SS.p_DA[n,t1] - SS.p_DA[n,t2] <= SS.u_comp27[n,t1]*SS.M1_comp27 )
            SS.comp27_4.add( SS.mu_min_R[n,t1] <= (1-SS.u_comp27[n,t1])*SS.M2_comp27 )
                         
    SS.comp28_1 = pe.ConstraintList()  
    SS.comp28_2 = pe.ConstraintList()
    SS.comp28_3 = pe.ConstraintList()
    SS.comp28_4 = pe.ConstraintList()
    for n in SS.N:
        SS.comp28_1.add( SS.Ramp[n]*SS.U_ini[n] + SS.p_DA[n,SS.T[1]] - SS.P_ini[n] >=0 )
        SS.comp28_2.add( SS.mu_min_R[n,SS.T[1]] >=0 )
        SS.comp28_3.add( SS.Ramp[n]*SS.U_ini[n] + SS.p_DA[n,SS.T[1]] - SS.P_ini[n] <= SS.u_comp28[n,SS.T[1]]*SS.M1_comp28  )
        SS.comp28_4.add( SS.mu_min_R[n,SS.T[1]] <= (1-SS.u_comp28[n,SS.T[1]])*SS.M2_comp28 )
    
    
    SS.comp29_1 = pe.ConstraintList()
    SS.comp29_2 = pe.ConstraintList()
    SS.comp29_3 = pe.ConstraintList()
    SS.comp29_4 = pe.ConstraintList()
    for t in SS.T:
        for g in SS.G:
           SS.comp29_1.add( SS.u_DA[g,t] >= 0 )
           SS.comp29_2.add( SS.mu_min_B[g,t] >= 0 )
           SS.comp29_3.add( SS.u_DA[g,t] <= SS.u_comp29[g,t]*SS.M1_comp29 )
           SS.comp29_4.add( SS.mu_min_B[g,t] <= (1-SS.u_comp29[g,t])*SS.M2_comp29 )
            
    SS.comp30_1 = pe.ConstraintList()
    SS.comp30_2 = pe.ConstraintList()
    SS.comp30_3 = pe.ConstraintList()
    SS.comp30_4 = pe.ConstraintList()
    for t in SS.T:
        for g in SS.G:
            SS.comp30_1.add( 1 - SS.u_DA[g,t] >= 0)
            SS.comp30_2.add( SS.mu_max_B[g,t] >= 0 )
            SS.comp30_3.add( 1 - SS.u_DA[g,t] <= SS.u_comp30[g,t]*SS.M1_comp30 )
            SS.comp30_4.add( SS.mu_max_B[g,t] <= (1-SS.u_comp30[g,t])*SS.M2_comp30 )
            
    
    ##KKT Self-scheduling  
    ##Stationarity conditions
    SS.L3_q_DA = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
            SS.L3_q_DA.add( - SS.lambda_DA_H[t] + SS.C_fuel[s]*SS.rho_heat[s] - SS.mu_min_H[s,t] + SS.mu_max_H[s,t]  + SS.mu_C[s,t]*SS.r_chp[s] - SS.mu_min_C[s,t]*SS.rho_heat[s] + SS.mu_max_C[s,t]*SS.rho_heat[s] == 0)       
    
    SS.L3_p_DA = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for s in SS.SSC:
            SS.L3_p_DA.add( - SS.lambda_DA_E[t1] + SS.C_fuel[s]*SS.rho_elec[s]  - SS.mu_C[s,t1] - SS.mu_min_C[s,t1]*SS.rho_elec[s] + SS.mu_max_C[s,t1]*SS.rho_elec[s] \
                            + SS.mu_max_R[s,t1]  - SS.mu_max_R[s,t2] + SS.mu_min_R[s,t2] - SS.mu_min_R[s,t1] == 0 )   
                            
    #t24
    SS.L3_p_DA1 = pe.ConstraintList()
    for s in SS.SSC:
        SS.L3_p_DA1.add( - SS.lambda_DA_E[SS.T[T]] + SS.C_fuel[s]*SS.rho_elec[s]  - SS.mu_C[s,SS.T[T]] - SS.mu_min_C[s,SS.T[T]]*SS.rho_elec[s] + SS.mu_max_C[s,SS.T[T]]*SS.rho_elec[s] \
                        + SS.mu_max_R[s,SS.T[T]] - SS.mu_min_R[s,SS.T[T]] == 0)   
                                         
    SS.L3_u_DA = pe.ConstraintList()
    for t1, t2 in zip(SS.T, SS.Tnot1):
        for s in SS.SSC:
            SS.L3_u_DA.add( SS.mu_min_C[s,t1]*SS.Fuel_min[s] - SS.mu_max_C[s,t1]*SS.Fuel_max[s] + SS.mu_max_SU[s,t1]*SS.C_SU[s] - SS.mu_max_SU[s,t2]*SS.C_SU[s] - SS.mu_min_B[s,t1] + SS.mu_max_B[s,t1] \
                           - SS.mu_max_R[s,t1]*SS.Ramp[s] - SS.mu_min_R[s,t2]*SS.Ramp[s] == 0 )
                                       
    #t24
    SS.L3_u_DA1 = pe.ConstraintList()
    for s in SS.SSC:
        SS.L3_u_DA1.add( SS.mu_min_C[s,SS.T[T]]*SS.Fuel_min[s] - SS.mu_max_C[s,SS.T[T]]*SS.Fuel_max[s] + SS.mu_max_SU[s,SS.T[T]]*SS.C_SU[s] - SS.mu_min_B[s,SS.T[T]] + SS.mu_max_B[s,SS.T[T]] \
                       - SS.mu_max_R[s,SS.T[T]]*SS.Ramp[s] == 0 )
                   
    SS.L3_c_DA = pe.ConstraintList()   
    for t in SS.T:
        for s in SS.SSC:
            SS.L3_c_DA.add( 1 - SS.mu_max_SU[s,t] - SS.mu_min_SU[s,t] == 0 )
            
    ##Complementarity conditions        
    SS.comp31_1 = pe.ConstraintList()
    SS.comp31_2 = pe.ConstraintList()
    SS.comp31_3 = pe.ConstraintList()
    SS.comp31_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
           SS.comp31_1.add( SS.q_DA[s,t] >= 0 )
           SS.comp31_2.add( SS.mu_min_H[s,t] >= 0 )
           SS.comp31_3.add( SS.q_DA[s,t] <= SS.u_comp31[s,t]*SS.M1_comp31 )
           SS.comp31_4.add( SS.mu_min_H[s,t] <= (1-SS.u_comp31[s,t])*SS.M2_comp31 )
            
    SS.comp32_1 = pe.ConstraintList()
    SS.comp32_2 = pe.ConstraintList()
    SS.comp32_3 = pe.ConstraintList()
    SS.comp32_4 = pe.ConstraintList()
    
    for t in SS.T:
        for s in SS.SSC:
            SS.comp32_1.add( SS.Q_max[s] - SS.q_DA[s,t] >= 0 )
            SS.comp32_2.add( SS.mu_max_H[s,t] >= 0 )
            SS.comp32_3.add( SS.Q_max[s] - SS.q_DA[s,t] <= SS.u_comp32[s,t]*SS.M1_comp32 )
            SS.comp32_4.add( SS.mu_max_H[s,t] <= (1-SS.u_comp32[s,t])*SS.M2_comp32 )
    
    SS.comp33_1 = pe.ConstraintList()
    SS.comp33_2 = pe.ConstraintList()
    SS.comp33_3 = pe.ConstraintList()
    SS.comp33_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
           SS.comp33_1.add( SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t] >= 0 )
           SS.comp33_2.add( SS.mu_C[s,t] >= 0 )
           SS.comp33_3.add( SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t] <= SS.u_comp33[s,t]*SS.M1_comp33 )
           SS.comp33_4.add( SS.mu_C[s,t] <= (1-SS.u_comp33[s,t])*SS.M2_comp33 )
            
    SS.comp34_1 = pe.ConstraintList()
    SS.comp34_2 = pe.ConstraintList()
    SS.comp34_3 = pe.ConstraintList()
    SS.comp34_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
            SS.comp34_1.add( SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] >= 0 )
            SS.comp34_2.add( SS.mu_min_C[s,t] >= 0 )
            SS.comp34_3.add( SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s] <= SS.u_comp34[s,t]*SS.M1_comp34  )
            SS.comp34_4.add( SS.mu_min_C[s,t] <= (1-SS.u_comp34[s,t])*SS.M2_comp34 )
    
    SS.comp35_1 = pe.ConstraintList()
    SS.comp35_2 = pe.ConstraintList()
    SS.comp35_3 = pe.ConstraintList()
    SS.comp35_4 = pe.ConstraintList()
    
    for t in SS.T:
        for s in SS.SSC:
           SS.comp35_1.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t]>= 0 )
           SS.comp35_2.add( SS.mu_max_C[s,t] >= 0 )
           SS.comp35_3.add( SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t] <= SS.u_comp35[s,t]*SS.M1_comp35 )
           SS.comp35_4.add( SS.mu_max_C[s,t] <= (1-SS.u_comp35[s,t])*SS.M2_comp35 )
    
    SS.comp36_1 = pe.ConstraintList()
    SS.comp36_2 = pe.ConstraintList()
    SS.comp36_3 = pe.ConstraintList()
    SS.comp36_4 = pe.ConstraintList()
         
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.SSC:
            SS.comp36_1.add( SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]) >= 0 )
            SS.comp36_2.add( SS.mu_max_SU[s,t1] >= 0 )
            SS.comp36_3.add( SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]) <= SS.u_comp36[s,t1]*SS.M1_comp36 )
            SS.comp36_4.add( SS.mu_max_SU[s,t1] <= (1-SS.u_comp36[s,t1])*SS.M2_comp36 )
          
    SS.comp37_1 = pe.ConstraintList()
    SS.comp37_2 = pe.ConstraintList()
    SS.comp37_3 = pe.ConstraintList()
    SS.comp37_4 = pe.ConstraintList()
         
    for s in SS.SSC:
        SS.comp37_1.add( SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]) >= 0 )
        SS.comp37_2.add( SS.mu_max_SU[s,SS.T[1]] >= 0 )
        SS.comp37_3.add( SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]) <= SS.u_comp37[s,SS.T[1]]*SS.M1_comp37  )
        SS.comp37_4.add( SS.mu_max_SU[s,SS.T[1]] <= (1-SS.u_comp37[s,SS.T[1]])*SS.M2_comp37 )
             
    SS.comp38_1 = pe.ConstraintList()
    SS.comp38_2 = pe.ConstraintList()
    SS.comp38_3 = pe.ConstraintList()
    SS.comp38_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
           SS.comp38_1.add( SS.c_DA[s,t] >= 0 )
           SS.comp38_2.add( SS.mu_min_SU[s,t] >= 0 )
           SS.comp38_3.add( SS.c_DA[s,t] <= SS.u_comp38[s,t]*SS.M1_comp38 )
           SS.comp38_4.add( SS.mu_min_SU[s,t] <= (1-SS.u_comp38[s,t])*SS.M2_comp38 )
    
    SS.comp39_1 = pe.ConstraintList()  
    SS.comp39_2 = pe.ConstraintList()
    SS.comp39_3 = pe.ConstraintList()
    SS.comp39_4 = pe.ConstraintList()
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.SSC:
            SS.comp39_1.add( SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA[s,t2] - SS.p_DA[s,t1] >=0 )
            SS.comp39_2.add( SS.mu_max_R[s,t1] >= 0 )
            SS.comp39_3.add( SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA[s,t2] - SS.p_DA[s,t1] <= SS.u_comp39[s,t1]*SS.M1_comp39 )
            SS.comp39_4.add( SS.mu_max_R[s,t1] <= (1-SS.u_comp39[s,t1])*SS.M2_comp39 )
                         
    SS.comp40_1 = pe.ConstraintList()   
    SS.comp40_2 = pe.ConstraintList() 
    SS.comp40_3 = pe.ConstraintList() 
    SS.comp40_4 = pe.ConstraintList()   
    
    for s in SS.SSC:
        SS.comp40_1.add( SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA[s,SS.T[1]] >=0 )
        SS.comp40_2.add( SS.mu_max_R[s,SS.T[1]] >= 0 )
        SS.comp40_3.add( SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA[s,SS.T[1]] <= SS.u_comp40[s,SS.T[1]]*SS.M1_comp40 )
        SS.comp40_4.add( SS.mu_max_R[s,SS.T[1]] <= (1-SS.u_comp40[s,SS.T[1]])*SS.M2_comp40 )
    
    SS.comp41_1 = pe.ConstraintList() 
    SS.comp41_2 = pe.ConstraintList()
    SS.comp41_3 = pe.ConstraintList()
    SS.comp41_4 = pe.ConstraintList()
        
    for t1, t2 in zip(SS.Tnot1, SS.T):
        for s in SS.SSC:
            SS.comp41_1.add( SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA[s,t1] - SS.p_DA[s,t2] >=0 )
            SS.comp41_2.add( SS.mu_min_R[s,t1] >=0 )
            SS.comp41_3.add( SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA[s,t1] - SS.p_DA[s,t2] <= SS.u_comp41[s,t1]*SS.M1_comp41 )
            SS.comp41_4.add( SS.mu_min_R[s,t1] <= (1-SS.u_comp41[s,t1])*SS.M2_comp41 )
                         
    SS.comp42_1 = pe.ConstraintList()   
    SS.comp42_2 = pe.ConstraintList() 
    SS.comp42_3 = pe.ConstraintList() 
    SS.comp42_4 = pe.ConstraintList() 
      
    for s in SS.SSC:
        SS.comp42_1.add( SS.Ramp[s]*SS.U_ini[s] + SS.p_DA[s,SS.T[1]] - SS.P_ini[s] >=0)
        SS.comp42_2.add( SS.mu_min_R[s,SS.T[1]] >=0 )
        SS.comp42_3.add( SS.Ramp[s]*SS.U_ini[s] + SS.p_DA[s,SS.T[1]] - SS.P_ini[s] <= SS.u_comp42[s,SS.T[1]]*SS.M1_comp42)
        SS.comp42_4.add( SS.mu_min_R[s,SS.T[1]] <= (1-SS.u_comp42[s,SS.T[1]])*SS.M2_comp42 )
    
    
    SS.comp43_1 = pe.ConstraintList()
    SS.comp43_2 = pe.ConstraintList()
    SS.comp43_3 = pe.ConstraintList()
    SS.comp43_4 = pe.ConstraintList()
    
    for t in SS.T:
        for s in SS.SSC:
           SS.comp43_1.add( SS.u_DA[s,t] >= 0 )
           SS.comp43_2.add( SS.mu_min_B[s,t] >= 0 )
           SS.comp43_3.add( SS.u_DA[s,t] <= SS.u_comp43[s,t]*SS.M1_comp43 )
           SS.comp43_4.add( SS.mu_min_B[s,t] <= (1-SS.u_comp43[s,t])*SS.M2_comp43 )
            
    SS.comp44_1 = pe.ConstraintList()
    SS.comp44_2 = pe.ConstraintList()
    SS.comp44_3 = pe.ConstraintList()
    SS.comp44_4 = pe.ConstraintList()
    for t in SS.T:
        for s in SS.SSC:
            SS.comp44_1.add( 1 - SS.u_DA[s,t] >= 0 )
            SS.comp44_2.add( SS.mu_max_B[s,t] >= 0 )
            SS.comp44_3.add( 1 - SS.u_DA[s,t] <= SS.u_comp44[s,t]*SS.M1_comp44 )
            SS.comp44_4.add( SS.mu_max_B[s,t] <= (1-SS.u_comp44[s,t])*SS.M2_comp44 )
    
                
    #El Balance
    SS.El_DA_bal=pe.ConstraintList()
    for t in SS.T:
        SS.El_DA_bal.add( sum(SS.p_DA[i,t] for i in SS.I) + sum(SS.w_DA[w,t] for w in SS.W)  - SS.D_E[t]  == 0 )
           
    #Heat Balance
    SS.Heat_DA_bal=pe.ConstraintList()
    for t in SS.T:
        SS.Heat_DA_bal.add( sum(SS.q_DA[s,t] for s in SS.CHP) - SS.D_H[t]  == 0 )
        
     ### OBJECTIVE        
    SS.obj=pe.Objective(expr=1)
    
    
    solver = po.SolverFactory('gurobi')
    solver.options['MIPgap'] = 0.05
    #solver.options['IntFeasTol'] = 1e-2
    #solver.options['FeasibilityTol'] = 1e-5
    #solver.options['OptimalityTol'] = 1e-2
    solver.options['Presolve'] = 2
    solver.options['NumericFocus'] = 1
    solver.options['MIPFocus'] = 3
    solver.options['Method'] = 2
    #solver.options['parallel'] = 1
    solver.options['Seed'] = 866
    if scen=='s9':
       solver.options['FeasibilityTol'] = 1e-3 
    if scen=='s28':
       solver.options['FeasibilityTol'] = 1e-3 
    #if scen=='s29':
    #   solver.options['FeasibilityTol'] = 1e-3   
    if scen=='s47':
       solver.options['FeasibilityTol'] = 1e-3 
    print("\n MILP Displaying Solution\n" + '-'*60)
    #results = solver.solve(SS, load_solutions=True, tee=True, symbolic_solver_labels=True)
    results = solver.solve(SS, load_solutions=True, tee=True, symbolic_solver_labels=True)
    #SS.pprint()
    #sends results to stdout
    results.write()
    print("\n MILP Displaying Solution\n" + '-'*60)
    ##Printing the variables results
    # =============================================================================
    q_CHP={}
    p={}
    u_DA={}
    for t in SS.T:
         for s in SS.CHP:
             #print('q_DA[',t,',',s,',]', pe.value(SS.q_DA[s,t]))
             q_CHP[s,t]=pe.value(SS.q_DA[s,t])
    for t in SS.T:
        for s in SS.CHP:
            # print('u1[',t,',',s,',]', pe.value(SS.u_comp1[s,t]))
             p[s,t]=pe.value(SS.p_DA[s,t])
    # for t in SS.T:
    #     for i in SS.I:
    #         print('p_DA[',t,',',i,',]', pe.value(SS.p_DA[i,t]))
    # for t in SS.T:
    #     for w in SS.W:
    #         print('w_DA[',t,',',w,',]', pe.value(SS.w_DA[w,t]))
    for t in SS.T:
        for i in SS.I:
            #print('u_DA[',t,',',i,',]', pe.value(SS.u_DA[i,t]))
            u_DA[i,t]=pe.value(SS.u_DA[i,t])
    # for t in SS.T:
    #     for i in SS.I:
    #         print('c_DA[',t,',',i,',]', pe.value(SS.c_DA[i,t]))
    lambda_DA_E_SS={}
    for t in SS.T:
        #print('price_DA_E[',t,']', pe.value(SS.lambda_DA_E[t]))
        lambda_DA_E_SS[t]=pe.value(SS.lambda_DA_E[t])
    lambda_DA_H_SS={}
    for t in SS.T:
        #print('price_DA_H[',t,']', pe.value(SS.lambda_DA_H[t]))
        lambda_DA_H_SS[t]=pe.value(SS.lambda_DA_H[t])    
    # =============================================================================
    # =============================================================================
    # total_cost_it={}
    # 
    # for t in SS.T:
    #     for s in SS.CHP:
    #         total_cost_it[s,t]=SS.C_fuel[s]*(SS.rho_heat[s]*pe.value(SS.q_DA[s,t])+SS.rho_elec[s]*pe.value(SS.p_DA[s,t])) + pe.value(SS.c_DA[s,t])        
    # for t in SS.T:
    #     for g in SS.G:
    #         total_cost_it[g,t]=SS.C_fuel[g]*pe.value(SS.p_DA[g,t]) + pe.value(SS.c_DA[g,t])        
    # #print(total_cost_it)
    # 
    # total_cost2 = sum(total_cost_it[s,t] for t in SS.T for s in SS.CHP) + sum(total_cost_it[g,t] for t in SS.T for g in SS.G)
    # print("Total_cost2= ", total_cost2)
    total_cost_SS=sum( SS.C_fuel[s]*(SS.rho_heat[s]*pe.value(SS.q_DA[s,t]) + SS.rho_elec[s]*pe.value(SS.p_DA[s,t])) + pe.value(SS.c_DA[s,t]) for t in SS.T for s in SS.CHP ) + sum(SS.C_fuel[g]*pe.value(SS.p_DA[g,t]) + pe.value(SS.c_DA[g,t]) for t in SS.T for g in SS.G)
    print("Total_cost_SS= ", total_cost_SS)
    # #total_cost1= sum(t, sum(s,    + c_DA.l(s,t)) + sum(g, idata(g,'C_fuel')*p_DA.l(g,t) +c_DA.l(g,t)  )   );
    cost_heat=sum( SS.C_fuel[s]*(SS.rho_heat[s]*pe.value(SS.q_DA[s,t])) + pe.value(SS.c_DA[s,t]) for t in SS.T for s in SS.CHP ) 
    print("Cost_heat= ", cost_heat)
    cost_elec=sum( SS.C_fuel[s]*(SS.rho_elec[s]*pe.value(SS.p_DA[s,t]))  for t in SS.T for s in SS.CHP ) + sum(SS.C_fuel[g]*pe.value(SS.p_DA[g,t]) + pe.value(SS.c_DA[g,t]) for t in SS.T for g in SS.G)
    print("Cost_elec= ", cost_elec) 
    profit_it={}
     
    for t in SS.T:
         for s in SS.CHP:
             profit_it[s,t]= pe.value(SS.p_DA[s,t])*(pe.value(SS.lambda_DA_E[t]) - SS.rho_elec[s]*SS.C_fuel[s]) + pe.value(SS.q_DA[s,t])*(pe.value(SS.lambda_DA_H[t]) - SS.rho_heat[s]*SS.C_fuel[s]) - pe.value(SS.c_DA[s,t])        
    for t in SS.T:
         for g in SS.G:
             profit_it[g,t]= pe.value(SS.p_DA[g,t])*(pe.value(SS.lambda_DA_E[t]) - SS.C_fuel[g]) - pe.value(SS.c_DA[g,t])        
    #print(profit_it)
    profit={}
    for i in SS.I:
         profit[i]=sum(profit_it[i,t] for t in SS.T)
    print('Profit of each unit')
    print(profit)
    # =============================================================================
    Wind_curt={}
    for t in SS.T:
        for w in SS.W:
            Wind_curt[w,t]=SS.Wind_DA[w,t] - pe.value(SS.w_DA[w,t])   
    Wind_curt_total=sum(Wind_curt[w,t] for t in SS.T for w in SS.W)
    #Check all the complemetarity conditions
    ccomp1=sum( pe.value( SS.q_DA[s,t]*SS.mu_min_H[s,t])  for t in SS.T for s in SS.NSS );
    ccomp2=sum( pe.value( (SS.Q_max[s] - SS.q_DA[s,t])*SS.mu_max_H[s,t] ) for t in SS.T for s in SS.NSS)
    ccomp3=sum( pe.value( (SS.p_DA_H[s,t]-SS.r_chp[s]*SS.q_DA[s,t])*SS.mu_C_H[s,t]) for t in SS.T for s in SS.NSS)
    ccomp4=sum( pe.value( (SS.rho_elec[s]*SS.p_DA_H[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s])*SS.mu_min_C_H[s,t] ) for t in SS.T for s in SS.NSS)
    ccomp5=sum( pe.value( (SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA_H[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t])*SS.mu_max_C_H[s,t] ) for t in SS.T for s in SS.NSS )
    ccomp6=sum( pe.value( (SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]))*SS.mu_max_SU[s,t1]) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.NSS )
    ccomp7=sum( pe.value( (SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]))*SS.mu_max_SU[s,SS.T[1]]) for s in SS.NSS)
    ccomp8=sum( pe.value( SS.c_DA[s,t]*SS.mu_min_SU[s,t]) for t in SS.T for s in SS.NSS  )
    ccomp9=sum( pe.value( (SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA_H[s,t2] - SS.p_DA_H[s,t1])*SS.mu_max_R_H[s,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.NSS  )
    ccomp10=sum(pe.value( (SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA_H[s,SS.T[1]])*SS.mu_max_R_H[s,SS.T[1]]) for s in SS.NSS)
    ccomp11=sum(pe.value( (SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA_H[s,t1] - SS.p_DA_H[s,t2])*SS.mu_min_R_H[s,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.NSS)
    ccomp12=sum(pe.value( (SS.Ramp[s]*SS.U_ini[s] + SS.p_DA_H[s,SS.T[1]] - SS.P_ini[s])*SS.mu_min_R_H[s,SS.T[1]] ) for s in SS.NSS)
    ccomp13=sum(pe.value( SS.u_DA[s,t]*SS.mu_min_B[s,t]  )  for t in SS.T for s in SS.NSS  )
    ccomp14=sum(pe.value( (1 - SS.u_DA[s,t])*SS.mu_max_B[s,t] ) for t in SS.T for s in SS.NSS  )
    
    ccomp15=sum(pe.value( SS.w_DA[w,t]*SS.mu_min_W[w,t]) for t in SS.T for w in SS.W  )
    ccomp16=sum(pe.value( (SS.Wind_DA[w,t] - SS.w_DA[w,t])*SS.mu_max_W[w,t] )  for t in SS.T for w in SS.W )
    ccomp17=sum(pe.value( (SS.p_DA[g,t] - SS.P_min[g]*SS.u_DA[g,t])*SS.mu_min_G[g,t] ) for t in SS.T for g in SS.G  )
    ccomp18=sum(pe.value( (SS.P_max[g]*SS.u_DA[g,t] - SS.p_DA[g,t])*SS.mu_max_G[g,t] ) for t in SS.T for g in SS.G )
    ccomp19=sum(pe.value( (SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t])*SS.mu_C[s,t]) for t in SS.T for s in SS.NSS) 
    ccomp20=sum(pe.value( (SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s])*SS.mu_min_C[s,t] ) for t in SS.T for s in SS.NSS  )
    ccomp21=sum(pe.value( (SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] - SS.rho_heat[s]*SS.q_DA[s,t])*SS.mu_max_C[s,t] ) for t in SS.T for s in SS.NSS )
    ccomp22=sum(pe.value( (SS.c_DA[g,t1] - SS.C_SU[g]*(SS.u_DA[g,t1] - SS.u_DA[g,t2]))*SS.mu_max_SU[g,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for g in SS.G )
    ccomp23=sum(pe.value( (SS.c_DA[g,SS.T[1]] - SS.C_SU[g]*(SS.u_DA[g,SS.T[1]] - SS.U_ini[g]))*SS.mu_max_SU[g,SS.T[1]]) for g in SS.G )
    ccomp24=sum(pe.value( SS.c_DA[g,t]*SS.mu_min_SU[g,t] ) for t in SS.T for g in SS.G )
    ccomp25=sum(pe.value( (SS.Ramp[n]*SS.u_DA[n,t1] + SS.p_DA[n,t2] - SS.p_DA[n,t1])*SS.mu_max_R[n,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for n in SS.N )
    ccomp26=sum(pe.value( (SS.Ramp[n]*SS.u_DA[n,SS.T[1]] + SS.P_ini[n] - SS.p_DA[n,SS.T[1]])*SS.mu_max_R[n,SS.T[1]] ) for n in SS.N )
    ccomp27=sum(pe.value( (SS.Ramp[n]*SS.u_DA[n,t2] + SS.p_DA[n,t1] - SS.p_DA[n,t2])*SS.mu_min_R[n,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for n in SS.N )
    ccomp28=sum(pe.value( (SS.Ramp[n]*SS.U_ini[n] + SS.p_DA[n,SS.T[1]] - SS.P_ini[n])*SS.mu_min_R[n,SS.T[1]] ) for n in SS.N )
    ccomp29=sum(pe.value( SS.u_DA[g,t]* SS.mu_min_B[g,t] ) for t in SS.T for g in SS.G)
    ccomp30=sum(pe.value( (1 - SS.u_DA[g,t])*SS.mu_max_B[g,t] ) for t in SS.T for g in SS.G)
    
    ccomp31=sum(pe.value( SS.q_DA[s,t]*SS.mu_min_H[s,t]  ) for t in SS.T for s in SS.SSC )
    ccomp32=sum(pe.value( (SS.Q_max[s] - SS.q_DA[s,t])*SS.mu_max_H[s,t] ) for t in SS.T for s in SS.SSC )
    ccomp33=sum(pe.value( (SS.p_DA[s,t]-SS.r_chp[s]*SS.q_DA[s,t])*SS.mu_C[s,t] ) for t in SS.T for s in SS.SSC )
    ccomp34=sum(pe.value( (SS.rho_elec[s]*SS.p_DA[s,t] + SS.rho_heat[s]*SS.q_DA[s,t] - SS.u_DA[s,t]*SS.Fuel_min[s])*SS.mu_min_C[s,t] ) for t in SS.T for s in SS.SSC )
    ccomp35=sum(pe.value( (SS.u_DA[s,t]*SS.Fuel_max[s] -  SS.rho_elec[s]*SS.p_DA[s,t] -  SS.rho_heat[s]*SS.q_DA[s,t])*SS.mu_max_C[s,t] ) for t in SS.T for s in SS.SSC )
    ccomp36=sum(pe.value( (SS.c_DA[s,t1] - SS.C_SU[s]*(SS.u_DA[s,t1] - SS.u_DA[s,t2]))*SS.mu_max_SU[s,t1]) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.SSC)
    ccomp37=sum(pe.value( (SS.c_DA[s,SS.T[1]] - SS.C_SU[s]*(SS.u_DA[s,SS.T[1]] - SS.U_ini[s]))*SS.mu_max_SU[s,SS.T[1]])  for s in SS.SSC )
    ccomp38=sum(pe.value(  SS.c_DA[s,t]*SS.mu_min_SU[s,t] ) for t in SS.T for s in SS.SSC )
    ccomp39=sum(pe.value( (SS.Ramp[s]*SS.u_DA[s,t1] + SS.p_DA[s,t2] - SS.p_DA[s,t1])*SS.mu_max_R[s,t1]) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.SSC)
    ccomp40=sum(pe.value( (SS.Ramp[s]*SS.u_DA[s,SS.T[1]] + SS.P_ini[s] - SS.p_DA[s,SS.T[1]])*SS.mu_max_R[s,SS.T[1]] ) for s in SS.SSC )
    ccomp41=sum(pe.value( (SS.Ramp[s]*SS.u_DA[s,t2] + SS.p_DA[s,t1] - SS.p_DA[s,t2])*SS.mu_min_R[s,t1] ) for t1, t2 in zip(SS.Tnot1, SS.T) for s in SS.SSC)
    ccomp42=sum(pe.value( (SS.Ramp[s]*SS.U_ini[s] + SS.p_DA[s,SS.T[1]] - SS.P_ini[s])*SS.mu_min_R[s,SS.T[1]] ) for s in SS.SSC )
    ccomp43=sum(pe.value( SS.u_DA[s,t]*SS.mu_min_B[s,t] ) for t in SS.T for s in SS.SSC)
    ccomp44=sum(pe.value( (1 - SS.u_DA[s,t])*SS.mu_max_B[s,t] ) for t in SS.T for s in SS.SSC )
    
    fig, ax = plt.subplots(figsize = (4, 3))
    q_chp={}
    for s in SS.CHP:
        q_t=[]
        for t in SS.T:
            q_t.append(pe.value(SS.q_DA[s,t]))
        q_chp[s]=q_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    
    ax.stackplot(indata.time_numbers, q_chp.values(),
                 labels=q_chp.keys(), alpha=0.8)
    #ax.set_title('Heat dispatch: SS')
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
    plt.savefig('Heat_SS.pdf', dpi=100, bbox_inches='tight')
    plt.show    
    #Electricity dispatch Integrated
    fig1, ax1 = plt.subplots(figsize = (4, 3))
    p_DA_all={}
    
    for i in SS.I:
        p_t=[]
        for t in SS.T:
            p_t.append(pe.value(SS.p_DA[i,t]))
        p_DA_all[i]=p_t
    for w in SS.W:
        p_t=[]
        for t in SS.T:
            p_t.append(pe.value(SS.w_DA[w,t]))
        p_DA_all[w]=p_t
    wind_SP=['W_SP']
    for w, w1 in zip(SS.W, wind_SP):
        p_t=[]
        for t in SS.T:
            p_t.append(SS.Wind_DA[w,t] - pe.value(SS.w_DA[w,t]))
        p_DA_all[w1]=p_t
    #ax.stackplot=(time, q_chp.values(),
    #              labels = q_chp.keys(), alpha=0.8)
    labels1=[str(), str(), 'G1', 'G2', 'W1', 'Wind spillage']
    ax1.stackplot(indata.time_numbers, p_DA_all.values(),
                 labels=labels1, alpha=0.8)
    #ax1.set_title('Power dispatch: SS')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Power [MW]') 
    ax1.set_ylim(0, 900)
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)
    ax1.legend(bbox_to_anchor=(1.17, 1.05), fancybox=True, shadow=True)   
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    #ax1.legend(loc='upper left')   
    E_DA={}
    for i in p_DA_all.keys():
        E_DA[i]=int(sum(p_DA_all[i]))
    ax1.text(8, 100, E_DA['CHP1'])
    ax1.text(8, 340, E_DA['CHP2'])
    ax1.text(8, 540, E_DA['G2'])
    ax1.text(8, 720, E_DA['W1'])
    ax1.text(3, 400, E_DA['W_SP'])
    plt.savefig('Power_SS.pdf', dpi=100, bbox_inches='tight')
    plt.show    
    
    #Dispatch and price
    fig4, ax4 = plt.subplots(figsize = (4, 3))
    #color = 'b'
    ax4.set_xlabel('Time Period')
    ax4.set_ylabel('Power [MW]')
    ax4.bar(indata.time_numbers,p_DA_all['CHP2'], fill=False, edgecolor='tab:blue', label='CHP2')
    ax4.set_ylim(0, 280)
    #ax4.legend()
    ax5 = ax4.twinx()
    #color = 'k'
    ax5.plot(list(lambda_DA_E_SS.values()), color='k', label='Electricity Price')
    ax5.set_ylabel('Price [EUR/MWh]')
    ax5.set_ylim(0, 80)
    #ax5.legend()
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax5.get_legend_handles_labels()
    ax4.legend(h1+h2, l1+l2, frameon=False)
    plt.xticks(np.arange(min(indata.time_numbers)-1, max(indata.time_numbers)+1, 3.0))
    fig4.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Power_Price_SS.pdf', dpi=100, bbox_inches='tight')
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
    plt.savefig('CHP_dispatch_SS_big.pdf', dpi=100, bbox_inches='tight')
    plt.show
    
    return Wind_curt_total, total_cost_SS, lambda_DA_E_SS, lambda_DA_H_SS, u_DA 

        
    
    

