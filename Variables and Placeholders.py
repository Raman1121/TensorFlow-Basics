
# coding: utf-8

# ## Variables and Placeholders

# In[1]:


import tensorflow as tf


# In[2]:


sess = tf.InteractiveSession()


# In[3]:


my_tensor = tf.random_uniform((4,4),0, 1)


# In[4]:


my_tensor


# In[7]:


my_var = tf.Variable(initial_value = my_tensor)


# In[8]:


print(my_var)


# ### If we run 'sess.run(my_var)' here, we will get an error because we have to initialize the variables.

# In[9]:


init = tf.global_variables_initializer()


# In[10]:


sess.run(init)


# In[11]:


sess.run(my_var)


# ### PlaceHolders

# In[12]:


ph = tf.placeholder(tf.float32)

