import wfdb # Library for loading the data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve
import os

# Function for loading data
def loadData(data_dir, recordID):
    # Loading a record
    recordName = os.path.join(data_dir,str(recordID))
    sigs, fields = wfdb.rdsamp(recordName)

    # Loading the annotation
    ann = wfdb.io.rdann(recordName,'atr')

    # Extracting beat locations and labels (normal vs. other)
    beatLoc = ann.sample[1:]
    tmp = ann.symbol[1:]
    idx = [ind for ind, ele in enumerate(tmp) if ele=='N' or ele=='.']
    beatLab = np.ones(len(beatLoc))
    beatLab[idx] = 0

    # Extracting the first signals
    sig = sigs[:,0]

    perAbnormal = 100*sum(beatLab)/len(beatLab)
    print('[Record {:d}] Normal = {:4.1f} % | Abnormal = {:4.1f} %'.format(recordID,
        100-perAbnormal,perAbnormal))

    return sig, beatLoc, beatLab


# Function for extracting windowed data
def getDataArr(sig,beatLoc,beatLab,winSzHalf):
    xArr = []
    yArr = []
    for k,loc in enumerate(beatLoc):
        if((loc>winSzHalf-1) and (loc<len(sig)-winSzHalf-1)):
            xArr.append(sig[loc-winSzHalf:loc+winSzHalf])
            yArr.append(beatLab[k])
    xArr = np.array(xArr)
    yArr = np.array(yArr)>0

    return xArr, yArr

# Function for visualizing the high-dimensional data
def tsne_plot(xData,yData):
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random',perplexity=15).fit_transform(xData)
    zDf = pd.DataFrame(data=X_embedded, columns=['z1', 'z2'])

    x = zDf['z1'][yData==0]
    y = zDf['z2'][yData==0]
    plt.scatter(x,y,color='b')
    x = zDf['z1'][yData==1]
    y = zDf['z2'][yData==1]
    plt.scatter(x,y,color='r')
    plt.xlabel('z1')
    plt.ylabel('z2')

# CNN-Based Encoder
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, winSzHalf):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(True)
        )

        intSz = int((winSzHalf/4))
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(intSz * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
# CNN-Based Decoder
class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, winSzHalf):
        super().__init__()

        intSz = int((winSzHalf/4))

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, intSz * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, intSz))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 5, stride=2, padding=2, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

# Function for training model for an epoch
def train_epoch(encoder, decoder, device, data, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    # Move tensor to the proper device
    data_batch = data.to(device)
    # Encode data
    encoded_data = encoder(data_batch)
    # Decode data
    decoded_data = decoder(encoded_data)
    # Evaluate loss
    loss = loss_fn(decoded_data, data_batch)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.detach().cpu().numpy()

    return train_loss

# Function for converting numpy batch to tensor
def get_tensor(xData):
    sz = xData.shape
    return torch.tensor(xData.reshape([sz[0],1,sz[1]]),dtype=torch.float)

# Function for doing a full training of a model
def train_model(encoder,decoder,device,loss_fn,optim,xTrain,num_epochs):
   xTrain_Tensor = get_tensor(xTrain)
   loss_hist = []

   for epoch in range(num_epochs):
      train_loss = train_epoch(encoder,decoder,device,xTrain_Tensor,loss_fn,optim)
      if(np.mod(epoch+1,50)==0):
         print('EPOCH {:3d}/{} \t train loss = {:.4f}'.format(epoch + 1, num_epochs,train_loss))
      loss_hist.append(train_loss)

   return loss_hist

# Function for evaluating the encode and decoder
def eval_encdec(encoder, decoder, device, xData):
    # Set evaluation mode for encoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Getting tensor
        xTensor = get_tensor(xData)
        # Move tensor to the proper device
        data_batch = xTensor.to(device)
        # Encode data
        encoded_data = encoder(data_batch)
        # Decode data
        decoded_data = decoder(encoded_data)

    enc_data = encoded_data.detach().cpu().numpy()
    dec_data = np.squeeze(decoded_data.detach().cpu().numpy())
    
    return enc_data, dec_data

# Generates an ROC curve
def plot_ROC(encoder,decoder,device,xData,yData):
    [z,xHat] = eval_encdec(encoder,decoder,device,xData)
    yScore = np.mean(np.power(xHat-xData,2),axis=1)

    fpr, tpr, thresholds = roc_curve(yData, yScore)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    auc = roc_auc_score(yData,yScore)

    return auc

# Function for splitting data into training and testing
def splitData(xArr,yArr,perTrain):
    idx = int(np.round(xArr.shape[0]*perTrain))
    xTrain = xArr[0:idx,:]
    yTrain = yArr[0:idx]
    xTest = xArr[idx:,]
    yTest = yArr[idx:]

    return xTrain,yTrain,xTest,yTest
