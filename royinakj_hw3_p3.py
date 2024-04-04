# %%
import numpy as np
import pandas as pd
import torch 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %% [markdown]
# ### Part 3.a.

# %%
files = ['./Shakespeare.txt', './WarAndPeace_Tolstoy.txt','./PrideAndPrejudice_JaneAusten.txt']#'./MurderOnLinks_Agatha.txt',
a=[]
for file in files:
    a = a +[open(file, encoding = 'utf-8').read()]
print(len(a), len(a[0]), len(a[1]), len(a[2]))

# %%
a[0][:100]

# %% [markdown]
# ##### One-hot-encoding using only numpy for a sample of the first book

# %%

characters = set(a)
token_index = dict(zip(characters, range(1, len(characters) + 1)))
results = np.zeros((len(a),  max(token_index.values()) + 1))
for i, sample in enumerate(a[:10]):
    print(sample)
    index = token_index.get(sample)
    results[i, index] = 1.
results[:10]
    

# %%
list(token_index.keys())[np.argmax(results[0])-1]

# %%
list(token_index.keys())[np.argmax(results[1])-1]

# %% [markdown]
# ##### Encoding using ordinal and one-hot encoder of sklearn - First create a vocab and fit the encoders

# %%
characters = set(a[0]).union(set(a[1])).union(set(a[2]))
len(list(characters)), list(characters)[:10]

# %%
%%time
### converting the strings to list of characters
list_char = []
for j in range(len(a)): ## iterating over 3 books
    one_book_char_list = []
    for i, sample in enumerate(a[j]):
        one_book_char_list.append(sample)
    list_char += [one_book_char_list]
len(list_char), len(list_char[0]), len(list_char[1]), len(list_char[2])

# %% [markdown]
# ##### OneHotEncoder and OrdinalEncoder are trained on the vocabulary

# %%
ord_enc = OrdinalEncoder()
vocab_y_ord = ord_enc.fit_transform(np.array(list(characters)).reshape(-1,1))
vocab_y_ord[:10], np.array(list(characters)).reshape(-1,1)[:10]

# %%
onehot_enc = OneHotEncoder(sparse_output=False, sparse=False)
vocab_y_onehot = onehot_enc.fit_transform(vocab_y_ord)
vocab_y_onehot.shape, vocab_y_onehot[0].shape, len(list(characters))

# %% [markdown]
# ##### Creating train and test split - 80/20 per book 

# %%
train_x = []
test_x = []
for i in range(len(list_char)):
    train_end_index = int(0.8*len(list_char[i]))
    train_x += [list_char[i][200:train_end_index]]
    test_x += [list_char[i][train_end_index:]]
    print(len(list_char[i]), len(train_x[i]), len(test_x[i]), len(train_x[i])/len(list_char[i]), len(test_x[i])/len(list_char[i]))

print(len(train_x), len(test_x))

# %%
4303215/598520 , 2582045/598520 

# %%
def get_embedding(text): ## text is a list of characters
    return np.squeeze(onehot_enc.transform(ord_enc.transform(np.array(text).reshape((-1,1)))))

# %%
%%time
## train dataset
### creating more batches of equal sequence length based on lowest len of the three books for train and test
seq_len = 32
min_seq_len = min(len(train_x[0]), len(train_x[1]), len(train_x[2]))
nbatch = min_seq_len//(seq_len+1) ## trimming data that overflows from batch size 32 for each seq
min_seq_len = nbatch*(seq_len+1)
final_train_x = []
for i in range(len(train_x)):
    for j in range(0, len(train_x[i]), min_seq_len):
        end_j = j+min_seq_len
        if end_j > len(train_x[i]): ## ignoring sequences that have length less that min_seq_len
            continue
        final_train_x += [get_embedding(train_x[i][j: end_j])] ## performing the encoding
len(final_train_x), len(final_train_x[0])

# %%
final_train_x = np.array(final_train_x)
final_train_x.shape, final_train_x[0,:2]

# %%
598290/33

# %%
n=10
final_train_x = final_train_x.reshape((final_train_x.shape[0]*n, int(final_train_x.shape[1]/((seq_len+1)*n))*(seq_len+1), final_train_x.shape[2]))
final_train_x.shape

# %%
%%time
## validation dataset
### creating more batches of equal sequence length based on lowest len of the three books for train and test
min_seq_len = min(len(test_x[0]), len(test_x[1]), len(test_x[2]))
nbatch = min_seq_len//(seq_len+1) ## trimming data that overflows from batch size 32 for each seq
min_seq_len = nbatch*(seq_len+1)
final_test_x = []
for i in range(len(test_x)):
    for j in range(0, len(test_x[i]), min_seq_len):
        end_j = j+min_seq_len
        if end_j > len(test_x[i]): ## ignoring sequences that have length less that min_seq_len
            continue
        final_test_x += [get_embedding(test_x[i][j: end_j])] ## performing the encoding
len(final_test_x), len(final_test_x[0])

# %%
final_test_x = np.array(final_test_x)
final_test_x.shape, final_test_x[0,:2]

# %%
np.isnan(final_train_x).sum()

# %%
np.isnan(final_test_x).sum()

# %%
def get_batch(source, i):
    data = source[:,i:i+seq_len, :]
    target = source[:,i+1:i+1+seq_len,:]
    return data, target

# %%
data, target = get_batch(final_train_x, 0) ## testing the get_batch_function
data.shape, target.shape

# %% [markdown]
# ### Part 3.b

# %%
import torch.nn as nn
import torch.nn.functional as F

# %%
### Defining the RNN model
class RNNModel(nn.Module):
    """Container module with a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity='tanh', batch_first = True)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device

# %%
hidden = 200
n_layers = 1
ntokens = len(characters)
model = RNNModel(ntokens, ntokens, hidden, n_layers).to(device)

# %%
import time
import math

# %%
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# %%
criterion = nn.NLLLoss()

# %%
bsz = final_train_x.shape[0]
def train(train_loss_list, optimizer, train_acc_list):
    # Turn on training mode which enables dropout.
    model.train()
    accuracy=0
    total_loss = 0.
    start_time = time.time()
    clip = 0.25 ##default
    hidden = model.init_hidden(bsz)
    for op_params in optimizer.param_groups:
        lr = op_params['lr']
    for batch, i in enumerate(range(0, len(final_train_x[0])-seq_len, seq_len)):

        # if batch % 4000 == 0 and batch>0:
        #    for op_params in optimizer.param_groups:
        #         op_params['lr'] = op_params['lr'] * 0.8
        #         lr = op_params['lr']

        data, targets = get_batch(final_train_x, i)
        data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        targets = torch.from_numpy(targets).type(torch.FloatTensor).to(device)
        _,labels=torch.max(targets.view(-1,ntokens),1)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()


        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        _,pred=torch.max(output.view(-1,ntokens),1)
        accuracy += (pred==labels).sum().item()
        

        total_loss += loss.item()
        optimizer.step()
        if batch % 50 == 0 and batch > 0:
            cur_loss = total_loss / 50
            elapsed = time.time() - start_time
            accuracy = accuracy/(len(labels)*50)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}| accuracy {:3f}'.format(
                epoch, batch, len(final_train_x) // seq_len, lr,
                elapsed * 1000 / 10, cur_loss, math.exp(cur_loss), accuracy))
            train_loss_list += [cur_loss]
            train_acc_list += [accuracy]
            total_loss = 0
            accuracy = 0
            start_time = time.time()
    return train_loss_list, train_acc_list


# %%
best_val_loss = None

# %%
def evaluate(data_source, eval_loss_list, eval_acc_list):
    # Turn on evaluation mode which disables dropout.
    bsz = data_source.shape[0]
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(bsz)
    accuracy=0
    with torch.no_grad():
        for i in range(0, len(data_source[0])- seq_len, seq_len):
            data, targets = get_batch(data_source, i)
            data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
            targets = torch.from_numpy(targets).type(torch.FloatTensor).to(device)
            _,labels=torch.max(targets.view(-1,ntokens),1)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = criterion(output, labels)
            total_loss += seq_len*bsz * loss.item()

            _,pred=torch.max(output.view(-1,ntokens),1)
            accuracy += (pred==labels).sum().item()

    accuracy = accuracy/((len(data_source[0])- seq_len)*bsz)
    eval_acc_list += [accuracy]
    total_loss = total_loss / ((len(data_source[0])- seq_len)*bsz)
    eval_loss_list += [total_loss]
    return total_loss, eval_loss_list, accuracy, eval_acc_list


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
epochs = 10
train_loss_list = []
eval_loss_list = []
train_acc_list = []
eval_acc_list = []
for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss_list, train_acc_list = train(train_loss_list, optimizer, train_acc_list)
        val_loss, eval_loss_list, accuracy, eval_acc_list = evaluate(final_test_x, eval_loss_list, eval_acc_list)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}| accuracy {:.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, accuracy))
        print('-' * 89)
        # # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0

# %%
np.save('train_loss_list_hw3_p3.npy', np.array(train_loss_list))
np.save('train_acc_list_hw3_p3.npy', np.array(train_acc_list))
np.save('eval_loss_list_hw3_p3.npy', np.array(eval_loss_list))
np.save('eval_acc_list_hw3_p3.npy', np.array(eval_acc_list))

# %%
len(train_loss_list), len(train_acc_list), len(eval_loss_list), len(eval_acc_list)

# %%
np.min(train_loss_list), np.max(train_acc_list) 

# %%
np.min(eval_loss_list), np.max(eval_acc_list) 

# %%
import matplotlib.pyplot as plt
plt.plot(train_loss_list)

# %%
plt.plot(train_acc_list)

# %%
plt.plot(eval_loss_list)

# %%
plt.plot(eval_acc_list)

# %%
pred_sent = 'o'
current_char = [pred_sent]
model.eval()
hidden = model.init_hidden(1)
embed = get_embedding(current_char)
for i in range(seq_len):
    data = torch.from_numpy(embed.reshape((1,1,-1))).type(torch.FloatTensor).to(device)
    output, hidden = model(data, hidden)
    _,pred=torch.max(output.view(-1,ntokens),1)
    embed = onehot_enc.transform(pred.cpu().numpy().reshape((-1,1)))
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)
    

# %%
pred_sent = 't'
current_char = [pred_sent]
model.eval()
hidden = model.init_hidden(1)
embed = get_embedding(current_char)
for i in range(seq_len):
    data = torch.from_numpy(embed.reshape((1,1,-1))).type(torch.FloatTensor).to(device)
    output, hidden = model(data, hidden)
    _,pred=torch.max(output.view(-1,ntokens),1)
    embed = onehot_enc.transform(pred.cpu().numpy().reshape((-1,1)))
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)
    

# %%
torch.save(model, 'finalmodel_hw3_p3.pt')

# %% [markdown]
# ##### When I store loss for each iteration instead of running mean over 50 iterations, the plot is as shown below

# %%
import matplotlib.pyplot as plt
plt.plot(train_loss_list)

# %%
plt.plot(eval_loss_list)

# %% [markdown]
# ### Part 3.c

# %%
%%time
## train dataset
### creating more batches of equal sequence length based on lowest len of the three books for train and test
seq_len = 128
min_seq_len = min(len(train_x[0]), len(train_x[1]), len(train_x[2]))
nbatch = min_seq_len//((seq_len)+1) ## trimming data that overflows from batch size 32 for each seq
min_seq_len = nbatch*(seq_len+1)
final_train_x = []
for i in range(len(train_x)):
    for j in range(0, len(train_x[i]),int(min_seq_len)):
        end_j = j+min_seq_len
        if end_j > len(train_x[i]): ## ignoring sequences that have length less that min_seq_len
            continue
        final_train_x += [get_embedding(train_x[i][j: end_j])] ## performing the ordinal encoding
len(final_train_x), len(final_train_x[0])

# %%
30*598273

# %%
final_train_x = np.array(final_train_x)
final_train_x.shape, final_train_x[0,:2]

# %%
598302/(129*6)

# %%
n=6
final_train_x = final_train_x.reshape((final_train_x.shape[0]*n, int(final_train_x.shape[1]/((seq_len+1)*n))*(seq_len+1), final_train_x.shape[2]))
final_train_x.shape

# %%
%%time
## validation dataset
### creating more batches of equal sequence length based on lowest len of the three books for train and test
min_seq_len = min(len(test_x[0]), len(test_x[1]), len(test_x[2]))
nbatch = min_seq_len//(seq_len) ## trimming data that overflows from batch size 32 for each seq
min_seq_len = nbatch*(seq_len) +1
final_test_x = []
for i in range(len(test_x)):
    for j in range(0, len(test_x[i]), min_seq_len):
        end_j = j+min_seq_len
        if end_j > len(test_x[i]): ## ignoring sequences that have length less that min_seq_len
            continue
        final_test_x += [get_embedding(test_x[i][j: end_j])] ## performing the ordinal encoding
len(final_test_x), len(final_test_x[0])

# %%
final_test_x = np.array(final_test_x)
final_test_x.shape, final_test_x[0,:2]

# %%
def get_batch(source, i):
    data = source[:,i:i+seq_len]
    target = source[:,i+1:i+1+seq_len]
    return np.concatenate([data, np.zeros((data.shape[0], data.shape[1], 5))], axis=2), np.concatenate([target, np.zeros((target.shape[0], target.shape[1], 5))], axis=2)

# %%
data, target = get_batch(final_train_x, 0) ## testing the get_batch_function
data.shape, target.shape

# %%
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# %%
class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        self.src_mask = mask
        # else:
        #     self.src_mask = None

        # src = self.input_emb(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        # output = output.view(-1, self.ntoken)
        return F.log_softmax(output, dim=-1)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device

# %%
n_layers = 4
ntokens = 128
hidden = 2*ntokens
nhead = 2
model = TransformerModel(ntokens, ntokens, nhead, hidden, n_layers).to(device)

# %%
import time
import math

# %%
criterion = nn.NLLLoss()

# %%
bsz = final_train_x.shape[0]
def train(train_loss_list, optimizer, train_acc_list):
    # Turn on training mode which enables dropout.
    model.train()
    accuracy=0
    total_loss = 0.
    start_time = time.time()
    clip = 1 ##default
    # hidden = model.init_hidden(bsz)
    for op_params in optimizer.param_groups:
        lr = op_params['lr']
    for batch, i in enumerate(range(0, len(final_train_x[0])-seq_len, seq_len)):

        # if batch % 4000 == 0 and batch>0:
        #    for op_params in optimizer.param_groups:
        #         op_params['lr'] = op_params['lr'] * 0.8
        #         lr = op_params['lr']

        data, targets = get_batch(final_train_x, i)
        data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        targets = torch.from_numpy(targets).type(torch.FloatTensor).to(device)
        targets = torch.permute(targets, (1, 0, 2))
        data = torch.permute(data, (1, 0, 2))
        # print(data.shape, targets.shape, targets.reshape(-1,ntokens).shape)
        _,labels=torch.max(targets.reshape(-1,ntokens),1)

        # data = torch.from_numpy(data).type(torch.IntTensor).to(device)
        # data = torch.permute(data, (1, 0, 2))
        # targets = torch.from_numpy(targets).type(torch.LongTensor).to(device)
        # targets = targets.view(-1)
        # targets = torch.permute(targets, (0,1))
        # targets = targets.view(-1,ntokens)
     
        output = model(data)
        # print(output.shape)
        output = output.reshape(-1, ntokens)
        # print(labels.shape, output.shape)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()


        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        _,pred=torch.max(output.reshape(-1,ntokens),1)
        accuracy += (pred==labels).sum().item()
        

        total_loss += loss.item()
        optimizer.step()
        if batch % 10 == 0 and batch > 0:
            cur_loss = total_loss / 10
            elapsed = time.time() - start_time
            accuracy = accuracy/(len(labels)*10)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}| accuracy {:3f}'.format(
                epoch, batch, len(final_train_x) // seq_len, lr,
                elapsed * 1000 / 10, cur_loss, math.exp(cur_loss), accuracy))
            train_loss_list += [cur_loss]
            train_acc_list += [accuracy]
            total_loss = 0
            accuracy = 0
            start_time = time.time()
    return train_loss_list, train_acc_list


# %%
best_val_loss = None

# %%
def evaluate(data_source, eval_loss_list, eval_acc_list):
    # Turn on evaluation mode which disables dropout.
    bsz = data_source.shape[0]
    model.eval()
    total_loss = 0.
    # hidden = model.init_hidden(bsz)
    accuracy=0
    with torch.no_grad():
        for i in range(0, len(data_source[0])- seq_len, seq_len):
            data, targets = get_batch(data_source, i)
            data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
            targets = torch.from_numpy(targets).type(torch.FloatTensor).to(device)
            targets = torch.permute(targets, (1, 0, 2))
            data = torch.permute(data, (1, 0, 2))
            # print(data.shape, targets.shape)
            _,labels=torch.max(targets.reshape(-1,ntokens),1)
            # data = torch.from_numpy(data).type(torch.IntTensor).to(device)
            # data = torch.permute(data, (1, 0, 2))
            # targets = torch.from_numpy(targets).type(torch.LongTensor).to(device)
            # targets = torch.permute(targets, (0,1))
            # targets = targets.view(-1,ntokens)
            # targets = targets.view(-1)
            output = model(data)
            output = output.reshape(-1, ntokens)
            loss = criterion(output, labels)
            total_loss += seq_len*bsz * loss.item()

            _,pred=torch.max(output.reshape(-1,ntokens),1)
            accuracy += (pred==labels).sum().item()

    accuracy = accuracy/((len(data_source[0])- seq_len)*bsz)
    eval_acc_list += [accuracy]
    total_loss = total_loss / ((len(data_source[0])- seq_len)*bsz)
    eval_loss_list += [total_loss]
    return total_loss, eval_loss_list, accuracy, eval_acc_list


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
epochs = 10
train_loss_list = []
eval_loss_list = []
train_acc_list = []
eval_acc_list = []
for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss_list, train_acc_list = train(train_loss_list, optimizer, train_acc_list)
        val_loss, eval_loss_list, accuracy, eval_acc_list = evaluate(final_test_x, eval_loss_list, eval_acc_list)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}| accuracy {:.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, accuracy))
        print('-' * 89)
        # # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0

# %%
### increasing amount of data points (taking more books as input)
### increase learning rate

# %%
np.save('train_loss_list_hw3_p3_transformer.npy', np.array(train_loss_list))
np.save('train_acc_list_hw3_p3_transformer.npy', np.array(train_acc_list))
np.save('eval_loss_list_hw3_p3_transformer.npy', np.array(eval_loss_list))
np.save('eval_acc_list_hw3_p3_transformer.npy', np.array(eval_acc_list))

# %%
len(train_loss_list), len(train_acc_list), len(eval_loss_list), len(eval_acc_list)

# %%
np.min(train_loss_list), np.max(train_acc_list) 

# %%
np.min(eval_loss_list), np.max(eval_acc_list) 

# %%
import matplotlib.pyplot as plt
plt.plot(train_loss_list)

# %%
plt.plot(train_acc_list)

# %%
plt.plot(eval_loss_list)

# %%
plt.plot(eval_acc_list)

# %%
from torch.distributions import Categorical

# %%
pred_sent = 'd'
current_char = [pred_sent]
model.eval()
embed = get_embedding(current_char)
embed = np.concatenate([embed, np.zeros((5,))], axis=0)
for i in range(seq_len):
    data = torch.from_numpy(embed.reshape((1,1,-1))).type(torch.FloatTensor).to(device)
    data = torch.permute(data, (1, 0, 2))
    output = model(data)
    # print(output[:,:,:123].shape)
    dist = Categorical(output[:,:,:123])
    index = dist.sample()
    # print(index)
    pred = index
    embed = onehot_enc.transform(pred.cpu().numpy().reshape((-1,1))).reshape(-1,)
    embed = np.concatenate([embed, np.zeros((5,))], axis=0)
    # print(embed.shape)
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)

# %%
pred_sent = 't'
current_char = [pred_sent]
model.eval()
embed = get_embedding(current_char)
embed = np.concatenate([embed, np.zeros((5,))], axis=0)
for i in range(seq_len):
    data = torch.from_numpy(embed.reshape((1,1,-1))).type(torch.FloatTensor).to(device)
    data = torch.permute(data, (1, 0, 2))
    output = model(data)
    _,pred = torch.max(output.reshape(-1,ntokens),1)
    embed = onehot_enc.transform(pred.cpu().numpy().reshape((-1,1))).reshape(-1,)
    embed = np.concatenate([embed, np.zeros((5,))], axis=0)
    # print(embed.shape)
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)

# %%
torch.save(model, 'finalmodel_hw3_p3_transformer.pt')

# %%
len('the world is beaut')

# %%
pred_sent = 'the world is bea'
one_book_char_list = []
for i, sample in enumerate(pred_sent):
    one_book_char_list.append(sample)
one_book_char_list

# %%
embed = get_embedding(one_book_char_list)
embed = np.concatenate([embed, np.zeros((embed.shape[0], 5))], axis=1)
embed.shape

# %%

pred_sent = 'the world is bea'
# one_book_char_list = []
# for i, sample in enumerate(pred_sent):
#     one_book_char_list.append(sample)
# embed = get_embedding(one_book_char_list)
# embed = np.concatenate([embed, np.zeros((embed.shape[0], 5))], axis=1)
for i in range(1024):
    one_book_char_list = []
    for j, sample in enumerate(pred_sent):
        one_book_char_list.append(sample)
    embed = get_embedding(one_book_char_list)
    embed = np.concatenate([embed, np.zeros((embed.shape[0], 5))], axis=1)
    data = torch.from_numpy(embed.reshape((1, embed.shape[0],embed.shape[1]))).type(torch.FloatTensor).to(device)
    data = torch.permute(data, (1, 0, 2))
    output = model(data)
    _,pred = torch.max(output[-1,:,:123].reshape(1,123),1)
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)

# %%
pred_sent = 'Sue and Johnsy lived at the top of a building with three floors. they discovered that they liked the same kind of art, the same kind of food'
for i in range(1024):
    one_book_char_list = []
    for j, sample in enumerate(pred_sent):
        one_book_char_list.append(sample)
    embed = get_embedding(one_book_char_list)
    embed = np.concatenate([embed, np.zeros((embed.shape[0], 5))], axis=1)
    data = torch.from_numpy(embed.reshape((1, embed.shape[0],embed.shape[1]))).type(torch.FloatTensor).to(device)
    data = torch.permute(data, (1, 0, 2))
    output = model(data)
    _,pred = torch.max(output[-1,:,:123].reshape(1,123),1)
    pred_sent +=ord_enc.inverse_transform(pred.cpu().numpy().reshape((-1,1)))[0][0]
print(pred_sent)


