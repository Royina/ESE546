#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision as thv
train = thv.datasets.MNIST('./', download=True, train=True)
val = thv.datasets.MNIST('./', download=True, train=False)
print('train dataset : ', train.data.shape, len(train.targets))
print('val dataset : ', val.data.shape, len(val.targets))


# In[2]:


## sampling 50% of train and test data
from sklearn.model_selection import train_test_split
X_train, _, y_train, _ = train_test_split(train.data.numpy(), train.targets.numpy(), stratify=train.targets.numpy(), 
                                                    test_size=0.5)
X_val, _, y_val, _ = train_test_split(val.data.numpy(), val.targets.numpy(), stratify=val.targets.numpy(), 
                                                    test_size=0.5)
X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

w = 10
h = 8
fig = plt.figure(figsize=(8, 10))
fig.tight_layout()
fig.subplots_adjust(hspace=.5)
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = X_train[i]
    ax = fig.add_subplot(rows, columns, i)
    ax.set_title('Label:'+str(y_train[i]), pad=2, size= 8)
    ax.imshow(img)
plt.show()


# In[4]:


class linear_t:
    def __init__(self) :
        # initialize to appropriate sizes, fill with Gaussian entries
        # normalize to make the Frobenius norm of w, b equal to 1
        mu = 0
        sigma = 0.1
        self.w = np.random.normal(mu, sigma, 10*784).reshape((10, 784))
        self.w = self.w / np.linalg.norm(self.w)
        self.b = np.random.normal(mu, sigma, (1,10))
        self.b = self.b / np.linalg.norm(self.b)
        self.dw = np.zeros((10,784))
        self.db = np.zeros((1,10))
    
    def forward(self,hl):
        hl1 = np.matmul(hl, self.w.T) + self.b
        # cache h^l in forward because we will need it to compute 
        # dw in backward
        self.hl = hl
        return hl1
    
    def backward(self, dhl1):
        dhl = np.matmul(dhl1, self.w)
        dw = np.matmul(dhl1.T, self.hl)
        db = dhl1.sum(axis=0).reshape((1,10))
        self.dw = dw
        self.db = db
        # notice that there is no need to cache dhl
        return dhl
    
    def zero_grad(self):
        # useful to delete the stored backprop gradients of the
        # previous mini-batch before you start a new mini-batch
        self.dw, self.db = 0*self.dw, 0*self.db
        


# In[11]:


## perturbing w to check dw
hl=np.random.randn(1,784)
l1=linear_t()
hl1=l1.forward(hl)
dhl1=np.zeros((1,10))

dhl1=np.zeros((1,10))
k=np.random.randint(0,10)
dhl1[0,k]=1
dhl=l1.backward(dhl1)

dw,db=l1.dw,l1.db

epsilon = np.zeros_like(l1.w)
epsilon[2,0] = np.random.randn()

pert_dw = ((np.matmul(hl, (l1.w+epsilon).T)) - (np.matmul(hl, (l1.w-epsilon).T)))/(2* epsilon[2,0])

print(dw[2,0], pert_dw[0,k])


# In[17]:


## db perturbation will yield same results as b is just directly added and subtracted
## dh perturbation
epsilon = np.zeros_like(hl)
epsilon[0,10] = np.random.randn()
pert_dh = ((np.matmul((hl+epsilon), l1.w.T)) - (np.matmul((hl-epsilon), l1.w.T)))/ (2* epsilon[0,10])

print(dhl[0,10], pert_dh[0,k])


# In[18]:


class relu_t:
    def __init__(self) :
        # no paramaters, nothing to initialize
        pass
    
    def forward(self,hl):
        hl1 = np.maximum(0, hl)
        # cache h^l in forward 
        self.hl = hl
        return hl1
    
    def backward(self, dhl1):
        dhl = np.where(self.hl>0, dhl1, 0).reshape(dhl1.shape)
        # print(dhl1, dhl)
        # notice that there is no need to cache dhl
        return dhl
    
    def zero_grad(self):
        pass


# In[19]:


a = np.random.normal(0, 0.1, 10)
print(a)
print(np.where(a>0,1,0).reshape((10)), a)


# In[20]:


class softmax_cross_entropy_t:
    def __init__(self):
        # no paramaters, nothing to initialize
        pass

    def forward(self, hl, y):
        ehl = np.exp(hl)
        hl1 = ehl/ ehl.sum(axis=1).reshape((32,1))
        self.hl1 = hl1
        self.y = y
        # compute average loss ell(y) over a mini-batch
        # print(np.sum(np.log(hl1) ,axis=0).shape)
        ell = (1/hl1.shape[0]) * (np.sum(-np.log(hl1)*y))
        # print(y.shape, y_pred.shape)
        error = (1/hl1.shape[0]) * ((np.argmax(y, axis=1)!=np.argmax(hl1, axis=1)).sum())
        return ell, error
    
    def backward(self):
        # as we saw in the notes, the backprop input to the 
        # loss layer is 1, so this function does not take any
        # arguments
        dhl = (1/self.hl1.shape[0])*(self.hl1 - self.y)
        return dhl
    
    def zero_grad(self):
        pass


# In[21]:


a=[1,1,3]
np.eye(10)[a], np.eye(10)[a].shape


# In[22]:


X_train = X_train.reshape((30000, 784))
X_train.shape


# In[23]:


X_train.max()


# In[24]:


X_train = X_train/255
X_train.shape, X_train.max()


# In[25]:


X_train.max()


# In[26]:


y_train = np.eye(10)[y_train]
y_train.shape


# In[27]:


y_train.shape


# In[35]:


# initialize all the layers
l1, l2, l3 = linear_t(), relu_t(), softmax_cross_entropy_t()
net = [l1, l2, l3]
lr = 0.2
error_list = []

# train for at least 1000 iterations
for t in range(50000):
    # 1. sample a mini-batch of size =32
    # each image in the mini - batch is chosen uniformly randomly from the
    # training dataset
    index = np.random.choice(30000, 32)
    x, y = X_train[index], y_train[index]

    # 2. zero gradient buffer
    for l in net:
        l.zero_grad()

    # 3. forward pass
    h1 = l1.forward(x)
    h2 = l2.forward(h1)
    ell , error = l3.forward(h2 , y)

    # 4. backward pass
    dh2 = l3.backward()
    dh1 = l2.backward(dh2)
    dx = l1.backward(dh1)

    # 5. gather backprop gradients 
    dw, db = l1.dw , l1.db

    # 6. print some quantities for logging 
    # and debugging
    print(t, error)
    error_list += [error]
    # print(l1.w, l1.b)
    # print(dw.shape,db)
    # print(t, np.linalg.norm(dw/l1.w), np.linalg.norm(db/l1.b))

    # 7. one step of SGD
    l1.w = l1.w - lr * dw
    l1.b = l1.b - lr * db


# In[43]:


plt.hist(error_list)


# In[45]:


plt.plot(error_list)


# In[44]:


np.mean(error_list), np.std(error_list)


# In[50]:


error_list[-10:] #last 10 epochs


# ### Running code on validation dataset

# In[52]:


X_val = X_val.reshape((-1, 784))
X_val = X_val/255
X_val.shape


# In[53]:


X_val.max()


# In[54]:


y_val = np.eye(10)[y_val]
y_val.shape


# In[62]:


def validate(w,b):
    # 1. iterate over mini-batches from the validation dataset
    # note that this should not be done randomly, we want to check
    # every image only once

    error_list = []
    ell_list = []
    loss, tot_error = 0, 0
    for i in range(0,5000,32):
        if 5000 < i+32:
            continue
        x, y = X_val[i:i+32,:], y_val[i:i+32, :]
        # compute forward pass and error
        h1 = l1.forward(x)
        h2 = l2.forward(h1)
        ell , error = l3.forward(h2 , y) 
        ell_list += [ell]
        error_list += [error]
        tot_error += error
        loss += ell

    return error_list, ell_list, loss, tot_error

error_list, ell_list, loss, tot_error = validate(l1.w, l1.b)
len(error_list), len(ell_list), loss, tot_error


# In[63]:


plt.plot(error_list) ## plot of validation error


# In[64]:


plt.plot(ell_list) ## plot of validation loss


# ### Implementing the same architecture with PyTorch

# In[179]:


import torch
from torch import nn


# In[180]:


X_train.max()


# In[181]:


from torch.utils.data import TensorDataset, DataLoader
tensor_x_train = torch.Tensor(X_train)
tensor_y_train = torch.Tensor(y_train)
tensor_x_val = torch.Tensor(X_val)
tensor_y_val = torch.Tensor(y_val)
tensor_x_train.shape, tensor_y_train.shape, tensor_x_val.shape, tensor_y_val.shape


# In[182]:


batch_size = 32

# Create data loaders.
train_dataloader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=batch_size)
test_dataloader = DataLoader(TensorDataset(tensor_x_val, tensor_y_val), batch_size=batch_size)


# In[183]:


# Get cpu or gpu device for training.
#device = "cuda" if torch.cuda.is_available() else "cpu"
device='cpu'
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 10),
            nn.ReLU(),
            
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# In[184]:


## optimizing the model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In[185]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct, error = 0, 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_t, current = loss.item(), batch * len(X)
            print(f"loss: {loss_t:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss += loss.item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        error += (pred.argmax(1) != y.argmax(1)).type(torch.float).sum().item()
    train_loss /= len(dataloader)
    correct /= size
    error /=size
    return (train_loss, error)


# In[186]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, error = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            error += (pred.argmax(1) != y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    error /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (test_loss, error)


# In[187]:


epochs = 10000
train_loss, train_error, test_loss, test_error = [], [], [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
    (loss, error) = train(train_dataloader, model, loss_fn, optimizer)
    train_loss += [loss]
    train_error += [error]
    (loss, error) = test(test_dataloader, model, loss_fn)
    test_loss += [loss]
    test_error += [error]
print("Done!")


# In[191]:


len(train_loss), len(train_error), len(test_loss), len(test_error)


# In[192]:


train_loss[-10:]


# In[193]:


## plotting train_loss
plt.plot(train_loss)


# In[194]:


## plotting train error
plt.plot(train_error)


# In[188]:


test_loss[-10:]


# In[189]:


## plotting test loss
plt.plot(test_loss)


# In[190]:


## plotting test loss
plt.plot(test_error)


# In[ ]:


## smaller number of epochs


# In[163]:


len(train_loss), len(train_error), len(test_loss), len(test_error)


# In[164]:


train_loss[-10:]


# In[165]:


## plotting train_loss
plt.plot(train_loss)


# In[166]:


## plotting train error
plt.plot(train_error)


# In[167]:


test_loss[-10:]


# In[168]:


## plotting test loss
plt.plot(test_loss)


# In[169]:


## plotting test error
plt.plot(test_error)

