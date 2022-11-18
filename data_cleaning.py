import pandas as pd
import numpy as np

import pickle


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

# read data
path = r"F:\File\CS151\project\allegation.xlsx"
# Allegations_raw = pd.read_excel(path,sheet_name=["Allegations"])["Allegations"]
Police_Witnesses_raw = pd.read_excel(path,sheet_name=["Police_Witnesses"])["Police_Witnesses"]
Complaining_Witnesses_raw = pd.read_excel(path,sheet_name=["Complaining_Witnesses"])["Complaining_Witnesses"]
Officer_Profile_raw = pd.read_excel(path,sheet_name=["Officer_Profile"])["Officer_Profile"]

Officer_Profile = Officer_Profile_raw.dropna(subset=['Age',"Race","Gender"])
Police_Witnesses  = Police_Witnesses_raw.dropna(subset=["Race","Gender"])
Complaining_Witnesses = Complaining_Witnesses_raw.dropna(subset=["Race","Gender"])

Officer_Profile = Officer_Profile.drop(['Unit','Star'],axis=1)
Complaining_Witnesses = Complaining_Witnesses.drop(['Age'],axis=1)

Police_Witnesses = Police_Witnesses[Police_Witnesses['Race']!='Unknown']
Officer_Profile = Officer_Profile[Officer_Profile['Race']!='Unknown']


save_variable(Officer_Profile , 'OP.txt')
save_variable(Police_Witnesses , 'PW.txt')
save_variable(Complaining_Witnesses , 'CW.txt')
