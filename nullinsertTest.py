import numpy as np
import pandas as pd
from datetime import datetime
import time 

temp = 1

def fake_dict(temp,datetime):
    fake = {
            "ID": temp,
            "time":datetime  
        }
    
    return fake



fake_lst = []

sec = 35
while temp<31:
    date = datetime.now()
    other_time = datetime(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute, second=sec)
    
    print(date.second)
    print(other_time.second)
    
    if  date.second == other_time.second:
        docker_fake = fake_dict(temp,date)
        sec += 3 
    else:
        docker_fake = fake_dict(temp,np.NAN)
        
    fake_lst.append(docker_fake)
    temp += 1
    time.sleep(1.0)
    # print("working",temp)

fake_df = pd.DataFrame(fake_lst)

print(fake_df)

start = time.time()
fake_df_na_deleted = fake_df.dropna()
print(fake_df_na_deleted)
print(time.time()- start)