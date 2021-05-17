#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
print(sklearn.__version__)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff


# In[4]:


data = arff.loadarff('E:\csc845\Scenario B\Scenario B\TimeBasedFeatures-Dataset-15s.arff')


# In[5]:


df = pd.DataFrame(data[0])


# In[20]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values.astype(str)


# In[21]:


y


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)


# In[25]:


y_pred = classifier.predict(x_test)


# In[26]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[27]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(15,8))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted',fontsize=18)
plt.ylabel('Truth',fontsize=18)


# In[52]:


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(y_test, y_pred)
plt.yticks(rotation = 90)
plt.ylabel('Predicted',fontsize=18)
plt.xlabel('Truth',fontsize=18)
plt.show()


# In[ ]:





# In[46]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

