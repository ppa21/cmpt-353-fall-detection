#!/usr/bin/env python
# coding: utf-8

# In[35]:


import sys
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import signal
import pickle
from tabulate import tabulate

def stats_generator(df):
    # total max Acceleration
    max_AT = df['AccelerationT'].idxmax()
    
    # 1 sec before and after drop data
    Imp_Data = df[max_AT - 50:max_AT + 25] 
    Imp_Data = pd.DataFrame(Imp_Data)
    
    MeanX = Imp_Data['AccelerationX'].mean()
    MeanY = Imp_Data['AccelerationY'].mean()
    MeanZ = Imp_Data['AccelerationZ'].mean()
    MeanT = Imp_Data['AccelerationT'].mean()
    
    StdX = Imp_Data['AccelerationX'].std()
    StdY = Imp_Data['AccelerationY'].std()
    StdZ = Imp_Data['AccelerationZ'].std()
    StdT = Imp_Data['AccelerationT'].std()
    
    MinX = Imp_Data['AccelerationX'].min()
    MinY = Imp_Data['AccelerationY'].min()
    MinZ = Imp_Data['AccelerationZ'].min()
    MinT = Imp_Data['AccelerationT'].min()
    
    MaxX = Imp_Data['AccelerationX'].max()
    MaxY = Imp_Data['AccelerationY'].max()
    MaxZ = Imp_Data['AccelerationZ'].max()
    MaxT = Imp_Data['AccelerationT'].max()
    
    stat = [{'MeanX':MeanX, 'MeanY':MeanY, 'MeanZ':MeanZ, 'MeanT':MeanT, 
             'StdX': StdX, 'StdY': StdY, 'StdZ': StdZ, 'StdT': StdT,
             'MinX': MinX, 'MinY': MinY, 'MinZ': MinZ, 'MinT': MinT,
             'MaxX':MaxX, 'MaxY':MaxY, 'MaxZ':MaxZ, 'MaxT':MaxT}]
    
    return pd.DataFrame(stat)

def main(in_directory):
    #opening the file
    # 1. Importing
    df = pd.read_csv(in_directory, sep = ';')
    
    # 2. Cleaning
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    low_passed_walkSitX = list(signal.filtfilt(b, a, df["AccelerationX"]))
    low_passed_walkSitY = list(signal.filtfilt(b, a, df["AccelerationY"]))
    low_passed_walkSitZ = list(signal.filtfilt(b, a, df["AccelerationZ"]))
    low_passed = pd.DataFrame(list(zip(low_passed_walkSitX, low_passed_walkSitY, low_passed_walkSitZ)),
                                            columns=["AccelerationX", "AccelerationY", "AccelerationZ"])
    low_passed["AccelerationT"] = np.sqrt(df["AccelerationX"]**2 * 
                                                     df["AccelerationY"]**2 *
                                                     df["AccelerationZ"]**2 )
    # 3. Transforming
    transformed = stats_generator(low_passed)
    
    # 4. Loding the model
    randomForestClf = pickle.load(open("Models/RandomForestClf.pkl", "rb"))
    rfResults = randomForestClf.predict(transformed)
    rfResultsP = randomForestClf.predict_proba(transformed)
             
    if rfResults:
        print("That was a fall !")
        print(f"- With an accuracy of { rfResultsP[0][1]*100} %")
    else:
        print("That was a not a fall !")
        print(f"- With an accuracy of { rfResultsP[0][0]*100} %")
 
    
if __name__ == '__main__':
    in_directory = sys.argv[1]
    main(in_directory)

