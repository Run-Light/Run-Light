#Import the libraries

import numpy as np
import math 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Validation.Validation import ExceptOneValidation

#Import and creation of a dataframe(Table)

data=pd.read_csv("data_subject_filtrage_5.csv")
df=data.copy()



def get_variation(df):
    '''
    df : DataFrame
    output : DataFrame with variations values in columns'''
    
    x_lum_var=df.x_lumbar-df.x_lumbar.shift(periods=1, fill_value=df.x_lumbar[len(df)-1])
    y_lum_var=df.y_lumbar-df.y_lumbar.shift(periods=1, fill_value=df.y_lumbar[len(df)-1])
    z_lum_var=df.z_lumbar-df.z_lumbar.shift(periods=1, fill_value=df.z_lumbar[len(df)-1])
    x_tib_var=df.x_tibia-df.x_tibia.shift(periods=1, fill_value=df.x_tibia[len(df)-1])
    y_tib_var=df.y_tibia-df.y_tibia.shift(periods=1, fill_value=df.y_tibia[len(df)-1])
    z_tib_var=df.z_tibia-df.z_tibia.shift(periods=1, fill_value=df.z_tibia[len(df)-1])

    df_var= pd.DataFrame(
    {'var_x_lumbar': x_lum_var,
     'var_y_lumbar': y_lum_var,
     'var_z_lumbar': z_lum_var,
     'var_x_tibia': x_tib_var,
     'var_y_tibia': y_tib_var,
     'var_z_tibia': z_tib_var
    })
    
    return df.join(df_var)

#Application of the function

df=get_variation(df)

#Drop columns we don't want/ are not really correlated 
X=df.drop(['index','nb_seance','nb_passage','x_force','z_force','y_force'],axis=1)
#Target
y=df['z_force']

#LGMB Regressor model 

lgbmr=LGBMRegressor(boosting_type='goss',random_state=42,n_estimators=300,learning_rate=0.2,n_jobs=-1)
score_lgbmr,error_lgbmr,predict_lgbmr= ExceptOneValidation(lgbmr,X,y)

#Results
print('MSE:',math.sqrt(pd.Series(error_lgbmr).mean()))
print('Score:',pd.Series(score_lgbmr).mean()*100,'%')

#Error 
print("The error summary for the model : ")
print(pd.Series(error_lgbmr).describe())



print("The score summary for the model : ")
print(pd.Series(score_lgbmr).describe())

#Plot preprocessing

df['z_pred']=predict_lgbmr
fig,axes = plt.subplots(20,1,figsize=(20,120))
for i in range(20):
    df['z_pred'].iloc[i*5000:i*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred'].iloc[(i+1)*5000:(i+1)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred'].iloc[(i+2)*5000:(i+2)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred'].iloc[(i+3)*5000:(i+3)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred'].iloc[(i+4)*5000:(i+4)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')

#Graph printing the results smooth : 

df['z_pred_smooth']=df['z_pred']
wind=201
windna=int((wind-1)/2)
for i in range(100):
    df['z_pred_smooth'].iloc[i*5000+windna:i*5000+5000-windna]=df['z_pred'].iloc[i*5000:i*5000+5000].rolling(wind,center=True).mean().iloc[windna:-windna]
    
fig,axes = plt.subplots(20,1,figsize=(20,120))
for i in range(20):
    df['z_pred_smooth'].iloc[i*5000:i*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred_smooth'].iloc[(i+1)*5000:(i+1)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred_smooth'].iloc[(i+2)*5000:(i+2)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred_smooth'].iloc[(i+3)*5000:(i+3)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_pred_{i}')
    df['z_pred_smooth'].iloc[(i+4)*5000:(i+4)*5000+5000].reset_index(drop=True).plot(ax=axes[i],title=f'z_prey_pred_smoothd_{i}')

#Comparison of the result (not smooth/smooth) :
  

print("MSE:")
print('Raw',mean_squared_error(df['z_force'],df['z_pred'])**(1/2),'Smooth:',mean_squared_error(df['z_force'],df['z_pred_smooth'])**(1/2))
print("Score:")
print('Raw',r2_score(df['z_force'],df['z_pred']),'Smooth:',r2_score(df['z_force'],df['z_pred_smooth']))

#Exportation of the predictions into a csv file :

np.savetxt("Exported_data\data_predict.csv", predict_lgbmr, delimiter=",")