import numpy as np
import pandas as pd
import statistics




def AVG_and_number(alist):
    AVGlist=[]
    for i in alist:
        AVGlist.append(i[1]-i[0])
        # print(i[0])
    TAD_average_size= statistics.mean(AVGlist)
    TAD_number=len(AVGlist)
    # print("AVG:",TAD_average_size)
    # print("NUMBER:",TAD_number)
    return  TAD_average_size,TAD_number
