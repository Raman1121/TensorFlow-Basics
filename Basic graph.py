
# coding: utf-8

# ## TensorFlow Graphs
# 

# In[1]:


import tensorflow as tf


# In[2]:


n1 = tf.constant(1)


# In[3]:


n2 = tf.constant(2)


# In[4]:


n3 = n1 + n2


# In[5]:


with tf.Session() as sess:
    result = sess.run(n3)


# In[6]:


print(result)


# ## Getting the default graph
# 

# In[7]:


print(tf.get_default_graph())


# ## Creating a new graph and setting it as the default graph

# In[10]:


graph_two = tf.Graph()


# In[14]:


with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

