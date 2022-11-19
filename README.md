# System for CS152_project
For CS 152 final project, I designed a `database` class, which can create and use database by `sqlite3`, clean and manage data by `pandas` and `numpy`, visualize by `matplotlib`.

When user makes query on table in database, the noise will be injected to the query result on demand. User can decide the amount of privacy budget and the column in table they want to add budget.

# Initialization example
## create database instance
```
    my_db = database()
```
## add table to database
```
    my_db.new_table(file_path+'/files/PW.txt',"Police_Witness")
    my_db.new_table(file_path+'/files/CW.txt',"Complaining_Witness")
    my_db.new_table(file_path+'/files/OP.txt',"Officer_Profile")
```
## check name of table in database
```
    my_db.tables_in_db()
```
## make a query
```
    res = my_db.read_sql_pd("SELECT Age,COUNT(*) FROM Officer_Profile GROUP BY Age",[1], 0.1)
    # The last 2 parameters mean a laplace noise with epsilon = 0.1 will be added into the second(index==1) column of query result
    print(res)
```
## check errors information of the last query
```
    my_db.print_info()
```

## visualization
```
    label=['20~30','30~40','40~50','50~60','60~70','70~80','80~90','90~100']
    my_db.make_pie(res["COUNT(*)"],label,"Age distribution in Officier Profile")
```
## compare number of errors and runtime over epsilon
```
    epi_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    query = "SELECT Age,COUNT(*) FROM Officer_Profile GROUP BY Age"
    my_db.plot_error_time_over_epi(epi_range,query,[1])
```
## check log
```
    print(my_db.log)
```
