#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This algorithm convert the matlab file where the data from the 3 running captors are stored: 
Stance force platform, Tibia captor and Lumba captor.
The output of the algorithm gives one csv with all the data of the captors.
To call the algorithm you have to call the load_conversion function with the file path.
"""

import pandas as pd
from scipy.io import loadmat

class conversion_matlab_csv:
    
    def conversion_tot(data,fil=4):
        '''
        Main function that call the other conversion for each captor and concat them in one data frame.
        And create a csv file with the data frame that will be stored in the current folder.

        Parameters
        ----------
        data : dictionary
            the matlab file load from the file path with the load_conversion function.
            
        fil : integer, optional
            The filtrage of the data in the dataset it's the first array in the matlab file and we only return one filtrage
            The default value is 4 because it's the filtrage with the best score it correspond to the 25Hz Cut-Off frequency.
        Returns
        -------
        df_tot : pandas.dataframe
            The final output with all the data in one dataframe with the sensors next to each other.

        '''
        stanceforce = data['StanceForce']  # variable in mat file
        stancetibia = data['StanceTibia']
        stancelumbar = data['StanceLumbar']
        df_Lumbar=conversion_matlab_csv.tab_Lumbar(stancelumbar,fil)
        df_Tibia=conversion_matlab_csv.tab_Tibia(stancetibia,fil)
        df_Force=conversion_matlab_csv.tab_Force(stanceforce,fil)
        df_tot=pd.concat([df_Lumbar,df_Tibia,df_Force],axis=1)
        df_tot.to_csv(f'data_filtrage_{fil+1}.csv',index=False)
        return df_tot

    def tab_Lumbar(stancelumbar,fil=4):
        '''
        Take every passage recorded with the lumabar captor and put them in 3 columns following the others passages.
        The 3 columns correspond to each axis (x,y,z).
        The count of seances and passages are incremented each time with 10 passages in each of the 10 seances.

        Parameters
        ----------
        stancelumbar : numpy.ndarray
            The column stancelumbar in the matlab dictionary.
        fil : integer, optional
            The filtrage of the data in the dataset it's the first array in the matlab file and we only return one filtrage
            The default value is 4 because it's the filtrage with the best score.

        Returns
        -------
        df_Lumbar : pandas.dataframe
            Dataframe with the data from the lumbar captors and the count of the seances and of the passages in 2 columns.

        '''
        df_Lumbar=pd.DataFrame(columns=['x_lumbar','y_lumbar','z_lumbar'])
        for i in range(len(stancelumbar[0][fil][0])):
            for j in range(len(stancelumbar[0][fil][0][0][0])):
                tampon=pd.DataFrame(stancelumbar[0][fil][0][i][0][j],columns=['x_lumbar','y_lumbar','z_lumbar'])
                tampon['nb_seance']=i
                tampon['nb_passage']=j
                df_Lumbar=pd.concat([df_Lumbar,tampon],axis=0)
            df_Lumbar=df_Lumbar.reindex(columns=['nb_seance','nb_passage','x_lumbar','y_lumbar','z_lumbar'])
            df_Lumbar.reset_index(inplace=True)
        return df_Lumbar

    def tab_Tibia(stancetibia,fil=4):
        '''
        Take every passage recorded with the tibia captor and put them in 3 columns following the others passages
        The 3 columns correspond to each axis (x,y,z).

        Parameters
        ----------
        stancetibia : numpy.ndarray
            The column stancelumbar in the matlab dictionary.
        fil : integer, optional
            The filtrage of the data in the dataset it's the first array in the matlab file and we only return one filtrage
            The default value is 4 because it's the filtrage with the best score.

        Returns
        -------
        df_Tibia : pandas.dataframe
            Dataframe with the data from the tibia captors.

        '''
        df_Tibia=pd.DataFrame(columns=['x_force','y_force','z_force'])
        for i in range(len(stancetibia[0][fil][0])):
            for j in range(len(stancetibia[0][fil][0][0][0])):
                tampon=pd.DataFrame(stancetibia[0][fil][0][i][0][j],columns=['x_tibia','y_tibia','z_tibia'])
                df_Tibia=pd.concat([df_Tibia,tampon],axis=0)
            df_Tibia=df_Tibia.reindex(columns=['x_tibia','y_tibia','z_tibia'])
            df_Tibia.reset_index(inplace=True,drop=True)
        return df_Tibia

    def tab_Force(stanceforce,fil=4):
        '''
        Take every passage recorded with the force platform captor and put them in 3 columns following the others passages
        The 3 columns correspond to each axis (x,y,z).

        Parameters
        ----------
        stanceforce : numpy.ndarray
            The column stanceforce in the matlab dictionary.
        fil : integer, optional
            The filtrage of the data in the dataset it's the first array in the matlab file and we only return one filtrage
            The default value is 4 because it's the filtrage with the best score.

        Returns
        -------
        df_Force : pandas.dataframe
            Dataframe with the data from the stance force plaform.

        '''
        df_Force=pd.DataFrame(columns=['x_force','y_force','z_force'])
        for i in range(len(stanceforce[0][fil][0])):
            for j in range(len(stanceforce[0][fil][0][0][0])):
                tampon=pd.DataFrame(stanceforce[0][fil][0][i][0][j],columns=['x_force','y_force','z_force'])
                df_Force=pd.concat([df_Force,tampon],axis=0)
            df_Force=df_Force.reindex(columns=['x_force','y_force','z_force'])
            df_Force.reset_index(inplace=True,drop=True)
        return df_Force

    def load_conversion(file_path,fil=4):
        '''
        The fonction to call with the matlab file path and the filtrage we want if we want an other one.
        Return a pandas.dataframe varible with all the data from the 3 captors.

        Parameters
        ----------
        file_path : file_path, text
            The path to acces the matlab file in order to run the conversion.
        fil : integer, optional
            The filtrage of the data in the dataset it's the first array in the matlab file and we only return one filtrage
            The default value is 4 because it's the filtrage with the best score.

        Returns
        -------
        df_total : pandas.dataframe
            The final output with all the data in one dataframe with the sensors next to each other

        '''
        data = loadmat(file_path)
        df_total=conversion_matlab_csv.conversion_tot(data,fil)
        return df_total