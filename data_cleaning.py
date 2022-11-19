import pandas as pd
import pickle
import os

file_path =os.path.dirname(__file__)


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
path = file_path+"/files/allegation.xlsx"
# Allegations_raw = pd.read_excel(path,sheet_name=["Allegations"])["Allegations"]
Police_Witnesses_raw = pd.read_excel(path,sheet_name=["Police_Witnesses"])["Police_Witnesses"]
Complaining_Witnesses_raw = pd.read_excel(path,sheet_name=["Complaining_Witnesses"])["Complaining_Witnesses"]
Officer_Profile_raw = pd.read_excel(path,sheet_name=["Officer_Profile"])["Officer_Profile"]

# Data cleaning
Officer_Profile = Officer_Profile_raw.dropna(subset=['Age',"Race","Gender"])
Police_Witnesses  = Police_Witnesses_raw.dropna(subset=["Race","Gender"])
Complaining_Witnesses = Complaining_Witnesses_raw.dropna(subset=["Race","Gender"])
Officer_Profile.loc[:,'Age']=Officer_Profile['Age'].apply(lambda x:x//10,)

Officer_Profile = Officer_Profile.drop(['Unit','Star'],axis=1)
Complaining_Witnesses = Complaining_Witnesses.drop(['Age'],axis=1)

Police_Witnesses = Police_Witnesses[Police_Witnesses['Race']!='Unknown']
Officer_Profile = Officer_Profile[Officer_Profile['Race']!='Unknown']

# Save data to .txt for later use
save_variable(Officer_Profile , file_path+'/files/OP.txt')
save_variable(Officer_Profile , file_path+'/files/CW.txt')
save_variable(Officer_Profile , file_path+'/files/PW.txt')
