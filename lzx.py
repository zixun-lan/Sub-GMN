from 数据生成 import data_generation1
import numpy as np

F = np.random.randint(0,10,size=9)
print(F)
data_generation1(core_size=9, whole_size=18, n=1100, F=F, self_loop=True, p=0.2, ppath='./0.2/train/')












