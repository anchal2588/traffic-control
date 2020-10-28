
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
cap_a=30
cap_b=35
a=np.random.randint(low=0,high=cap_a,size=(50),dtype='i')
b=np.random.randint(low=0,high=cap_b,size=(50),dtype='i')
c=np.random.randint(low=0,high=1,size=(50))
intersection_time=1.5
max_cycle_time=75
for i in range(0,len(a)):
    if(a[i]+b[i]<50):
        c[i]=(a[i]+b[i])*intersection_time
    else:
        c[i]=max_cycle_time
df=pd.DataFrame(list(zip(a,b,c)))
df.columns=['north-south','east-west','green_light']
print(df)


# In[2]:


tar=df.pop("green_light")


# In[3]:


mf=[[['gaussmf',{'mean':np.mean(np.arange(0,13)),
                'sigma':np.std(np.arange(0,13))}],
    ['gaussmf',{'mean':np.mean(np.arange(10,22)),
               'sigma':np.std(np.arange(10,22))}],
    ['gaussmf',{'mean':np.mean(np.arange(19,30)),
               'sigma':np.std(np.arange(19,30))}]],
    [['gaussmf',{'mean':np.mean(np.arange(0,14)),
                'sigma':np.std(np.arange(0,14))}],
    ['gaussmf',{'mean':np.mean(np.arange(11,26)),
               'sigma':np.std(np.arange(11,26))}],
    ['gaussmf',{'mean':np.mean(np.arange(22,35)),
               'sigma':np.std(np.arange(22,35))}]]]


# In[4]:


from membership import membershipfunction
mfc=membershipfunction.MemFuncs(mf)


# In[5]:


import anfis
anf=anfis.ANFIS(df,tar,mfc)
pred=anf.trainHybridJangOffLine(epochs=20)


# In[7]:



anf.plotErrors()


# In[8]:


anf.plotResults()

