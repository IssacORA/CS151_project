from sqlite3 import connect
# from new_main import my_system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import collections
import heapq
import random
import math
import pickle
from scipy import stats
import re
import time

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

class table:
    def __init__(self,table):
        self.table = table
        self.description = ''
        
    def load_variable(self,filename):
        f=open(filename,'rb')
        r=pickle.load(f)
        f.close()
        return r
    
class tables:
    def __init__(self):
        self.tables = {}
        
    def load_table(self,table,table_name):
        if table_name in self.tables:
            print("This table name has been used")
        else:
            self.tables[table_name] = table
            
class database:
    def __init__(self):
        self.conn = connect("my_db")
        self.eps=0.1
        self.Gamma = 0
        self.last_query_run_time = 0
        self.last_query_error = 0

# ------------table operation------------
    def new_table(self,table,table_name_in_db):
        tables = self.read_sql_pd('SELECT name FROM SQLITE_MASTER')
        
        if table_name_in_db in list(tables['name']):
            print("This name has been in the database!")
        else:
            table.to_sql(table_name_in_db,self.conn)
    
    def tables_in_db(self):
        tables = self.read_sql_pd('SELECT name FROM SQLITE_MASTER',[])

        print("table names:",list(tables['name'][::2]))
# ------------query operation------------
    def read_sql(self,sql_command):
        # read a sql command then execute it
        cur = self.conn.cursor()
        cur.execute(sql_command)
        return cur.fetchmany(3)
    
    def read_sql_pd(self,sql_command,privacy_lst): 
        # read a sql command and returns a pandas dataframe
        # You shoud make a list of cols which should not be protected.
        time_1 = time.time()
        res = pd.read_sql(sql_command,self.conn)   
        # print(res)    

        for col_n in privacy_lst:
            col_name = res.columns[col_n]
            col = res.iloc[:,col_n]
            sample = col[0]

            if isinstance(sample,str):
                # print("column {} is being applied GRR".format(col_name))
                # then add GRR
                res[col_name] = self.GRR(col)
            else:
                # print("column {} is being applied laplace machanism".format(col_name))
                # then add laplace noise
                # print("1111",res)
                
                res[col_name] = self.laplace(col) 
        time_2 = time.time()
        self.last_query_run_time = time_2-time_1
        # print(res)
        return res
               
# ------------noise injection------------

    def GRR(self,col):
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
        std = self.epi2std(self.eps)
        noise = np.random.normal(loc=0, scale=std, size=len(array))
        error=0
        for num in noise:
            error+=abs(num)

        # print("error:",error)
        
        # new = array.copy()
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
                my_db.read_sql_pd(query,cols)
                errors[-1].append(self.last_query_error)
                runtimes[-1].append(self.last_query_run_time)
        plt.subplot(1,2,1)
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
        plt.subplot(1,2,2)
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
        # plt.xticks([i+1 for i in range(len(errors))],labels = runtimes)
        
        plt.tight_layout()

        plt.show()
                
    def make_pie(self,count,label,title):
        plt.pie(x = count,labels = label ,autopct='%1.1f%%')
        plt.title(title)
        plt.show()  
# test_sys = my_system()
if __name__=="__main__":
    t1 = table(load_variable('PW.txt'))
    t2 = table(load_variable('CW.txt'))
    t3 = table(load_variable('OP.txt'))

    my_tables = tables()
    table_lst = [t1.table,t2.table,t3.table]

    for i in range(len(table_lst)):
        my_tables.load_table(table_lst[i],"t"+str(i+1))

    my_db = database()
    # my_db.new_table(t1.table,"t1")
    # my_db.new_table(t2.table,"t2")
    # my_db.read_sql("DROP TABLE t3")
    # my_db.new_table(t3.table,"t3")
    # my_db.tables_in_db()
    # t1 : Police Witness
    # t2 : Complaint Witness
    # t3 : Officer Profile
    
    my_db.eps=0.2
    res_1 = my_db.read_sql_pd("SELECT Race,COUNT(*) FROM t3 GROUP BY Race",[1])
    print(res_1)
    # my_db.make_pie(res_1["COUNT(*)"],res_1["Race"],"Race distribution among officers")

    # res_1 = my_db.read_sql_pd("SELECT Race,COUNT(*) FROM t3 GROUP BY Race",[1])
    # print(res_1)
    