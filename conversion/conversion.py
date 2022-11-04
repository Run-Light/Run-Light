#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:46:00 2022

@author: max
"""

import pandas as pd

def conversion_tot(data,fil=10):
    stanceforce = data['StanceForce']  # variable in mat file
    stancetibia = data['StanceTibia']
    stancelumbar = data['StanceLumbar']
    df_Lumbar=tab_Lumbar(stancelumbar,fil)
    df_Tibia=tab_Tibia(stancetibia,fil)
    df_Force=tab_Force(stanceforce,fil)
    df=pd.concat([df_Lumbar,df_Tibia,df_Force],axis=1)
    df.to_csv(f'data_filtrage_{fil+1}.csv',index=False)
    return df

def tab_Lumbar(stancelumbar,fil=10):
    df_Lumbar=pd.DataFrame(columns=['x_lumbar','y_lumbar','z_lumbar'])
    for i in range(10):
        for j in range(len(stancelumbar[0][fil][0][0][0])):
            tampon=pd.DataFrame(stancelumbar[0][fil][0][i][0][j],columns=['x_lumbar','y_lumbar','z_lumbar'])
            tampon['nb_seance']=i
            tampon['nb_passage']=j
            df_Lumbar=pd.concat([df_Lumbar,tampon],axis=0)
        df_Lumbar=df_Lumbar.reindex(columns=['nb_seance','nb_passage','x_lumbar','y_lumbar','z_lumbar'])
        df_Lumbar.reset_index(inplace=True)
    return df_Lumbar

def tab_Tibia(stancetibia,fil=10):
    df_Tibia=pd.DataFrame(columns=['x_force','y_force','z_force'])
    for i in range(10):
        for j in range(len(stancetibia[0][fil][0][0][0])):
            tampon=pd.DataFrame(stancetibia[0][fil][0][i][0][j],columns=['x_tibia','y_tibia','z_tibia'])
            df_Tibia=pd.concat([df_Tibia,tampon],axis=0)
        df_Tibia=df_Tibia.reindex(columns=['x_tibia','y_tibia','z_tibia'])
        df_Tibia.reset_index(inplace=True,drop=True)
    return df_Tibia

def tab_Force(stanceforce,fil=10):
    df_Force=pd.DataFrame(columns=['x_force','y_force','z_force'])
    for i in range(10):
        for j in range(len(stanceforce[0][fil][0][0][0])):
            tampon=pd.DataFrame(stanceforce[0][fil][0][i][0][j],columns=['x_force','y_force','z_force'])
            df_Force=pd.concat([df_Force,tampon],axis=0)
        df_Force=df_Force.reindex(columns=['x_force','y_force','z_force'])
        df_Force.reset_index(inplace=True,drop=True)
    return df_Force

