
import rpy2.robjects as robjects 
from rpy2.robjects import pandas2ri
import pandas as pd     
import matplotlib.pyplot as plt                                                                         
pandas2ri.activate() 

readRDS = robjects.r['readRDS']                                                                                                  

group_participation= readRDS('data_EcoGrid/group_participation_september_on.rds')  
group_participation= pandas2ri.ri2py(group_participation) 
group_participation['house_number']=group_participation['house_number'].astype(int)
group_participation['group']=group_participation['group'].astype(int)

all_group= readRDS('data_EcoGrid/parameters.rds')  
all_group=  pandas2ri.ri2py(all_group) 
all_group =all_group.set_index('to_ts',append=True) 
 
power_consumption = readRDS('data_EcoGrid/power_consumption_september_to_feb.rds')
power_consumption = pandas2ri.ri2py(power_consumption) 
power_consumption =power_consumption.set_index('to_ts',append=True)  
