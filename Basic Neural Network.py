
# coding: utf-8

# ## Basic Neural Network

# In[29]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[5]:


n_features  = 10
n_dense_neurons = 3


# In[6]:


x = tf.placeholder(tf.float32, (None, n_features))


# In[7]:


w = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

b = tf.Variable(tf.ones([n_dense_neurons]))


# In[9]:


xW = tf.matmul(x,w)


# In[11]:


z = tf.add(xW, b)


# In[12]:


a = tf.sigmoid(z)


# In[13]:


init = tf.global_variables_initializer()


# In[16]:


with tf.Session() as sess:
    sess.run(init)
    
    layer_out = sess.run(a, feed_dict = {x : np.random.random([1, n_features])})


# In[17]:


print(layer_out)


# ## Simple Regression Example

# In[21]:


x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)


# In[22]:


y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)


# In[30]:


plt.plot(x_data, y_label)


# In[31]:


### y = mx + b


# In[32]:


### Initialize with some random values


# In[33]:


m = tf.Variable(0.44)
b = tf.Variable(0.87)


# In[36]:


error = 0

for x,y in zip(x_data, y_label):
    
    y_hat = y*x + b
    
    error = error + (y - y_hat)**2


# In[37]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)


# In[38]:


init = tf.global_variables_initializer()


# In[40]:


with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 1;
    
    for i in range(training_steps):
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])


# In[41]:


x_test = np.linspace(-1, 11, 10)

#y = m*x + b

y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')

