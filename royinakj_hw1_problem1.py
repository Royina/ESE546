#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install scikit-learn scikit-image')


# In[4]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ds = fetch_openml ('mnist_784', as_frame = False )

x, x_test , y, y_test = train_test_split(ds.data , ds.target ,test_size =0.2, random_state =42)


# In[ ]:


x.shape


# In[ ]:


import matplotlib.pyplot as plt
a = x[0]. reshape((28 ,28))
plt. imshow(a)


# In[ ]:


y.shape


# In[ ]:


ds.data.shape


# In[ ]:


ds.target.shape


# In[ ]:


import pandas as pd
df = pd.DataFrame({'Images':list(ds.data), 'Target': ds.target})


# In[ ]:


import matplotlib.pyplot as plt
a = df.Images[0]. reshape((28 ,28))
plt. imshow(a)


# In[ ]:


df.Target.value_counts()


# In[ ]:


## sampling 1000 from each class
new_df = []
for category in df.Target.unique():
    new_df += [df[df.Target==category].sample(1000)]
print(len(new_df))
new_df = pd.concat(new_df)
new_df.reset_index(drop=True, inplace=True)
new_df.shape


# In[ ]:


type(new_df.Images[0])


# In[ ]:


## downsizing the images to 14 by 14
import cv2
new_df['Final_Img'] = new_df.Images.apply(lambda x: cv2.resize(x.reshape((28,28)),(14,14)).flatten())
plt.imshow(new_df.Final_Img[1111].reshape(14,14))


# In[ ]:


new_df.head(1)


# In[ ]:


new_df.Final_Img[0].shape


# In[ ]:


import numpy as np
x, x_test , y, y_test = train_test_split(np.stack(new_df.Final_Img) , new_df.Target ,test_size =0.2, random_state =42)
x.shape


# In[ ]:


plt.imshow(x[0].reshape((14,14)))


# In[ ]:


y.shape


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC(gamma='auto', C=1, kernel = 'rbf')
clf.fit(x, y)


# In[ ]:


clf.score(x, y)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[ ]:


y_train_pred = clf.predict(x)
cm = confusion_matrix(y, y_train_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()


# In[ ]:


clf.score(x_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()


# In[ ]:


support_df = pd.Series(y).value_counts()
support_df = support_df.sort_index(ascending=True)
support_df = support_df.reset_index().rename(columns={'count':'total_train_samples'})
support_df2 = pd.Series(clf.n_support_).reset_index()
support_df2 = support_df2.rename(columns={0:'n_support_vectors'})
support_df = pd.concat([support_df, support_df2], axis=1)
support_df.drop(columns=['index'], inplace=True)
support_df


# In[ ]:


support_df['Ratio'] = support_df['n_support_vectors'] / support_df['total_train_samples']
support_df


# In[ ]:


### different svc paramteres - changing C
from sklearn.svm import SVC
clf = SVC(gamma='auto', C=0.5, kernel = 'linear')
clf.fit(x, y)


# In[ ]:


clf.score(x, y)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_train_pred = clf.predict(x)
cm = confusion_matrix(y, y_train_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()


# In[ ]:


clf.score(x_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Implementing Grid Search CV on svm.SVC

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'kernel':['linear', 'rbf'], 
              'C':[0.25, 0.5, 0.75, 1, 10],
              'gamma': ['scale', 'auto']}
svc = SVC()
clf_gs = GridSearchCV(svc, parameters)
clf_gs.fit(x, y)


# In[ ]:


result_df = pd.DataFrame(clf_gs.cv_results_)
print(result_df.shape)
result_df.head(20)


# In[ ]:


clf_gs.best_score_, clf_gs.best_params_


# ### Gabor filter

# In[ ]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ds = fetch_openml ('mnist_784', as_frame = False )


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
df = pd.DataFrame({'Images':list(ds.data), 'Target': ds.target})


# In[ ]:


import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# In[ ]:


train_idx = []
val_idx = []
for category in df.Target.unique():
    train_idx += df[df.Target==category].sample(100).index.to_list()
    val_idx += df[((~(df.index.isin(train_idx)))&(df.Target==category))].sample(100).index.to_list()
train_df = df[df.index.isin(train_idx)]
val_df = df[df.index.isin(val_idx)]
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
print(train_df.shape, val_df.shape)
train_df.Target.value_counts(), val_df.Target.value_counts()


# In[ ]:


train_df['Final_Img'] = train_df.Images.apply(lambda x: cv2.resize(x.reshape((28,28)),(14,14)).flatten())
val_df['Final_Img'] = val_df.Images.apply(lambda x: cv2.resize(x.reshape((28,28)),(14,14)).flatten())
x = np.stack(train_df.Final_Img)
y = train_df.Target
x_test = np.stack(val_df.Final_Img)
y_test = val_df.Target
train_df.shape, x.shape, y.shape, val_df.shape, x_test.shape, y_test.shape


# In[ ]:


plt.imshow(x[0].reshape((14,14)))


# In[ ]:


del train_df
del val_df
del train_idx
del val_idx


# In[ ]:


from skimage.filters import gabor_kernel, gabor


# In[ ]:


from skimage.filters import gabor_kernel, gabor


# In[ ]:


freq, theta, bandwidth = 0.1, np.pi/4, 1
gk = gabor_kernel(frequency = freq, theta = theta, bandwidth = bandwidth)
plt.figure(1); plt.clf(); plt.imshow(gk.real)
plt.figure(2); plt.clf(); plt.imshow(gk.imag)

# convolve the input image with the kernel and get co-efficients
# we will use only the real part and throw away the imaginary
# part of the co-efficients
image = x[0].reshape((14,14))
coeff_real, _ = gabor(image, frequency=freq, theta = theta,
                      bandwidth = bandwidth)
plt.figure(1); plt.clf(); plt.imshow(coeff_real)


# In[ ]:


image = x[0].reshape((14,14))
image.shape


# In[ ]:


freq, theta, bandwidth = 0.05, 0, 0.3
gk = gabor_kernel(frequency = freq, theta = theta, bandwidth = bandwidth)
coeff_real, _ = gabor(image, frequency=freq, theta = theta,
                      bandwidth = bandwidth)
coeff_real.shape


# In[ ]:


theta = np.arange(0, np.pi, np.pi/4)
frequency = np.arange(0.05, 0.5, 0.15)
bandwidth = np.arange(0.3, 1, 0.3)
theta, frequency, bandwidth


# In[ ]:


196*36+196


# In[ ]:


np.save('x_train.npy', x)
np.save('y_train.npy', y)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_gabor = []\nx_test_gabor = []\ncount=0\nfor i in range(x.shape[0]):\n    count += 1\n    input = []\n    t_input = []\n    image = x[i].reshape((14,14))\n    t_image = x_test[i].reshape((14,14))\n    for t in theta:\n        for f in frequency:\n            for b in bandwidth:\n                # print(t,f,b)\n                gk = gabor_kernel(frequency = f, theta = t, bandwidth = b)\n                coeff_real, _ = gabor(image, frequency=f, theta = t,\n                      bandwidth = b)\n                input += [coeff_real.reshape((196))]\n\n                t_coeff_real, _ = gabor(t_image, frequency=f, theta = t,\n                      bandwidth = b)\n                t_input += [t_coeff_real.reshape((196))]\n    x_gabor += [input]\n    x_test_gabor += [t_input]\n\n    if count%100 == 0:\n      print(count)\n')


# In[ ]:


x_gabor = np.array(x_gabor).reshape((x.shape[0], -1))
x_test_gabor = np.array(x_test_gabor).reshape((x.shape[0], -1))
x_gabor.shape, x_test_gabor.shape


# In[ ]:


x_test_gabor[0,:]


# In[ ]:


x_gabor[0,:]


# In[ ]:


np.save('Gabor_train_x_samples.npy', x_gabor)
np.save('Gabor_train_y_samples.npy', y)
np.save('Gabor_test_x_samples.npy', x_test_gabor)
np.save('Gabor_test_y_samples.npy', y_test)


# In[ ]:


### try scaling the input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x_gabor)
X_test = scaler.transform(x_test_gabor)
X.shape, X_test.shape


# In[ ]:


print(X.shape, X[827][:10])
print(X_test.shape, X_test[827][:10])


# In[ ]:


print(X.shape, X[0][:10])
print(X_test.shape, X_test[0][:10])


# In[ ]:


parameters = {'kernel':['linear', 'rbf'],
              'C':[0.25, 0.5, 0.75, 1, 10],
              'gamma': ['scale', 'auto']}
svc = SVC()
clf_gs = GridSearchCV(svc, parameters)
clf_gs.fit(X, y)


# In[ ]:


pd.DataFrame(clf_gs.cv_results_)


# In[ ]:


clf_gs.best_score_, clf_gs.best_params_


# In[ ]:


clf = SVC(gamma='scale', C=10, kernel = 'rbf')
clf.fit(X, y)


# In[ ]:


clf.score(X, y)


# In[ ]:


y_train_pred = clf.predict(X)
cm = confusion_matrix(y, y_train_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:


y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()

