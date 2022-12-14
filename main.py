from sqlite3 import connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import random
import pickle
from scipy import stats
import time
import math

file_path =os.path.dirname(__file__) # The directory of current file
            
class database:
    def __init__(self):
        self.conn = connect("my_db")
        self.eps=0
        self.Gamma = 0
        self.last_query_run_time = 0
        self.last_query_error = 0

# ------------table operation------------
    def load_variable(self,file_add):
        f=open(file_add,'rb')
        r=pickle.load(f)
        f.close()
        return r
    
    def new_table(self,file_add,table_name_in_db):
        table = self.load_variable(file_add)
        tables = self.read_sql_pd('SELECT name FROM SQLITE_MASTER',[],-1)
        
        if table_name_in_db in list(tables['name']):
            print("Table `{}` has been in the database!".format(table_name_in_db))
        else:
            table.to_sql(table_name_in_db,self.conn)
    
    def tables_in_db(self):
        tables = self.read_sql_pd('SELECT name FROM SQLITE_MASTER',[],-1)
        print("table names:",list(tables['name'][::2]))
        
# ------------query operation------------
    def read_sql(self,sql_command):
        # read a sql command then execute it
        cur = self.conn.cursor()
        cur.execute(sql_command)
        return cur.fetchmany(3)
    
    def read_sql_pd(self,sql_command,privacy_lst,eps): 
        # read a sql command and returns a pandas dataframe
        # You shoud make a list of cols which should not be protected.
        time_1 = time.time()
        res = pd.read_sql(sql_command,self.conn)   
  
        if eps!=-1:
            print('after if')
            self.eps=eps
            for col_n in privacy_lst:
                col_name = res.columns[col_n]
                col = res.iloc[:,col_n]
                sample = col[0]

                if isinstance(sample,str):
                    # then add GRR
                    res[col_name] = self.GRR(col)
                else:
                    # then add laplace noise
                    res[col_name] = self.laplace(col) 
        time_2 = time.time()
        self.last_query_run_time = time_2-time_1
        # print(res)
        return res
               
# ------------noise injection------------

    def GRR(self,col):
        print("GRRing")
        error=0
        p = self.Gamma+0.5
        unique = col.unique()
        unique_set = set(unique)
        rest = {}
        for elem in unique:
            res = unique_set - {elem}
            rest[elem] = list(res)
    
        def helper(elem):
            bernoulliDist = stats.bernoulli(p) 
            trails = bernoulliDist.rvs(1) 
            if trails==0:
                error+=1
                return random.sample(rest[elem],1)[0]
            else:
                return elem
        self.error = error
        return col.apply(helper)
            
    def laplace(self,array):
        print("laplacing")
        std = self.epi2std(self.eps)
        noise = np.random.normal(loc=0, scale=std, size=len(array))
        error=0
        for num in noise:
            error+=abs(num)

        self.last_query_error=error
        def no_neg(num):
            return 0 if num < 0 else num
        return list(map(no_neg,array+noise))
        
    def epi2std(self,eps):
        return 2/(eps**2)
    
# ------------analysis tool------------
        
    def plot_error_time_over_epi(self,epi_range,query,cols):
        errors=[]
        runtimes = []
        for eps in epi_range:
            self.eps = eps
            errors.append([])
            runtimes.append([])
            for _ in range(10):
                my_db.read_sql_pd(query,cols,self.eps)
                errors[-1].append(self.last_query_error)
                runtimes[-1].append(self.last_query_run_time)
        plt.subplot(1,3,1)
        plt.boxplot(errors, positions=[(i+1)*2 for i in range(len(errors))], widths=1.5, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})
        plt.xticks([(i+1)*2 for i in range(len(errors))],labels = epi_range)
        plt.xlabel("epsilon")
        plt.ylabel("number of errors")
        
        plt.subplot(1,3,2)
        log_error = errors.copy()
        for i in range(len(log_error)):
            for j in range(len(log_error[0])):
                log_error[i][j] = math.log(errors[i][j])
        

        plt.boxplot(log_error, positions=[(i+1)*2 for i in range(len(log_error))], widths=1.5, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})
        plt.xticks([(i+1)*2 for i in range(len(log_error))],labels = epi_range)
        plt.xlabel("epsilon",)
        plt.ylabel("log(number of errors)")
        # plt.yticks(rotation=90)
        
        
        plt.subplot(1,3,3)
        plt.boxplot(runtimes, positions=[(i+1)*2 for i in range(len(errors))], widths=1.5, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})
        plt.xticks([(i+1)*2 for i in range(len(errors))],labels = epi_range)
        plt.xlabel("epsilon")
        plt.ylabel("runtime")


        plt.show()
                
    def make_pie(self,count,label,title):
        plt.pie(x = count ,autopct='%1.1f%%')
        plt.title(title)
        plt.legend(label,loc='upper left')
        plt.tight_layout()
        plt.show()  
    
    def make_bar(self,count,label,title,xlabel,ylabel):
        plt.bar(x= label,height=count)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def make_count_pair(self,col):
        hashmap=Counter(col)
        count,label=[],[]
        for key in hashmap:
            label.append(key)
            count.append(hashmap[key])
        return count,label
    
    def print_info(self):
        print("epsilon=",my_db.eps,"number of error=",int(my_db.last_query_error),"variance=",round(my_db.epi2std(my_db.eps),2))

if __name__=="__main__":
    
    my_db = database()

    my_db.new_table(file_path+'/files/PW.txt',"Police_Witness")
    my_db.new_table(file_path+'/files/CW.txt',"Complaining_Witness")
    my_db.new_table(file_path+'/files/OP.txt',"Officer_Profile")
