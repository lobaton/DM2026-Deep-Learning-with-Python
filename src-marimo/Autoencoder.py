import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import wfdb
    import os
    from sklearn.manifold import TSNE
    import pandas as pd
    from torch import nn
    from sklearn.metrics import roc_auc_score, roc_curve

    return TSNE, mo, nn, np, os, pd, plt, roc_auc_score, roc_curve, torch, wfdb


@app.cell(hide_code=True)
def _(TSNE, nn, np, os, pd, plt, roc_auc_score, roc_curve, torch, wfdb):
    # HELPER FUNCTIONS

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

    return (
        Decoder,
        Encoder,
        eval_encdec,
        getDataArr,
        loadData,
        plot_ROC,
        train_model,
        tsne_plot,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Identifying Anomalies in ECG Data

    This notebook goes over a simple demonstration of how to use an autoencoder for anomlay detection. We make use of the [MIT-BIH Arrhytmia Database](https://physionet.org/content/mitdb/1.0.0/). The database contains ECG signals with beats individually labeled. Details can be found [here](https://physionet.org/files/mitdb/1.0.0/mitdbdir/intro.htm).

    Our goal will be detect any beats that are not labeled as normal. We will do this in an unsupervised way by training an AutoEncoder (AE) with a Convolutional Neural Network (CNN) architecture. The data will be split into windows around each heartbeat. We will label those cases that cannot be properly reconstructed as anomalies, and compare that with the actual labels from the data.
    """)
    return


@app.cell
def _(os, wfdb):
    # Loading the data. It will take a few minutes.

    data_dir = "mitdb-data"
    if os.path.isdir(data_dir):
        print(f"Data directory `{data_dir}` already exists.")
    else:
        wfdb.dl_database("mitdb", dl_dir=data_dir)
    return (data_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Visualization Without Encoding

    The following slider specified how big of a window (in samples) is used for the analysis. Try picking a window size that captures timing information of heartbeats.
    """)
    return


@app.cell
def _(mo):
    # Slider controlling half of the size of the window used for analysis
    winSzHalf_slider = mo.ui.slider(50, 800, step=10, value=400, label="Half window size", debounce=True)
    winSzHalf_slider
    return (winSzHalf_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Displaying the signal, a single waveform and the non-linear projection of the data to 2D using t-SNE. Note the impact of window size at trying to separate what is normal vs. abnormal.
    """)
    return


@app.cell
def _(data_dir, getDataArr, loadData, plt, tsne_plot, winSzHalf_slider):

    # Specifying a list of records to process. We are picking a subset because some
    # of the records have more non-normal beats than normal (e.g., in the case that
    # a pacemaker is present).
    recList = [100, 101, 103, 105, 108, 115, 116, 202, 205, 210, 121, 122,
                123, 112, 113, 117, 215, 219, 220, 230, 234]

    # Half of the size of the window used for analysis
    winSzHalf = winSzHalf_slider.value

    # Loading one of the records
    [_sig, _beatLoc, _beatLab] = loadData(data_dir,recList[0])

    sig = _sig
    beatLoc = _beatLoc
    beatLab = _beatLab

    # Converting continuous stream into windows around each heartbeat
    _xArr, _yArr = getDataArr(_sig,_beatLoc,_beatLab,winSzHalf)

    # Plotting the data
    _f = plt.figure()
    _f.set_figwidth(18)
    _f.set_figheight(5)

    # Plotting the signals and labels
    plt.subplot(1,3,1)
    plt.plot(_sig)
    _beatNormal = _beatLoc[_beatLab==0]
    plt.plot(_beatNormal,_sig[_beatNormal],'ks')
    _beatAbnormal = _beatLoc[_beatLab==1]
    plt.plot(_beatAbnormal,_sig[_beatAbnormal],'rs')
    plt.xlabel('time')
    plt.ylabel('ECG signal')
    plt.xlim([0, 2500])
    plt.ylim([-1,1.5])
    plt.grid()
    plt.title('Displaying Raw Single')

    # Plotting a single observation window
    plt.subplot(1,3,2)
    plt.plot(_xArr[0,])
    plt.xlabel('time')
    plt.ylabel('ECG signal')
    plt.grid()
    plt.title('Displaying a Single Observation Window')

    # Doing a scatter plot visualization of the raw data
    plt.subplot(1,3,3)
    tsne_plot(_xArr,_yArr)
    plt.axis('equal')
    plt.title('t-SNE Visualization')
    plt.legend(['Normal','Abnormal'])

    plt.gca()
    return recList, winSzHalf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training an AE a Single Record

    In this section, we train a simple CNN-based AE on a single record. We don't worry about verifying this using a testset since our target there is not to validate on the reconstruction, but to use the pipeline as a way to identify anomalies.

    The slider below specifies the dimensions of the latent variable `z`, i.e., it is the size of the bottleneck.
    """)
    return


@app.cell
def _(mo):
    # Slider for setting the dimension of the latent space
    d_latent_slider = mo.ui.slider(4, 128, step=2, value=32, label="Dimension of Latent Space", debounce=True)
    d_latent_slider
    return (d_latent_slider,)


@app.cell
def _(Decoder, Encoder, d_latent_slider, torch, winSzHalf):
    # Defining the loss function
    loss_fn = torch.nn.MSELoss()

    # Specifying the learning rate
    lr= 0.001

    # Setting a seed for random initialization
    torch.manual_seed(42)

    # Specifying the dimensions for the latent space
    d_latent = d_latent_slider.value

    # Initializing encoder, decoder and optimizer
    encoder = Encoder(d_latent,winSzHalf)
    decoder = Decoder(d_latent,winSzHalf)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device);
    decoder.to(device);
    return decoder, device, encoder, loss_fn, optim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following are the summaries of the encoder and decoder architecture selected. Note their symmetry.
    """)
    return


@app.cell
def _(encoder):
    encoder
    return


@app.cell
def _(decoder):
    decoder
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we select an specific record for training. Start with 0, and try 9 and 17 as well.
    """)
    return


@app.cell
def _(mo, recList):
    # Slider for selecting the Record ID for the data
    rec_id_slider = mo.ui.slider(0, len(recList)-1, step=1, value=0, label="Record ID to Train", debounce=True)
    rec_id_slider
    return (rec_id_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pressing the train button triggers the training of the model. This will take a few minutes.
    """)
    return


@app.cell
def _(mo):
    # Button used to trigger (re)training of the model on demand
    train_button = mo.ui.run_button(label="Train")
    train_button
    return (train_button,)


@app.cell
def _(
    data_dir,
    decoder,
    device,
    encoder,
    eval_encdec,
    getDataArr,
    loadData,
    loss_fn,
    mo,
    optim,
    recList,
    rec_id_slider,
    train_button,
    train_model,
    winSzHalf,
):
    # Skip this cell unless the Train button was pressed
    mo.stop(not train_button.value, mo.md("Click **Train** to train the model."))

    # Loading one of the records
    rec_id = rec_id_slider.value
    [_sig, _beatLoc, _beatLab] = loadData(data_dir,recList[rec_id])

    # Converting continuous stream into windows around each heartbeat
    xArr, yArr = getDataArr(_sig,_beatLoc,_beatLab,winSzHalf)

    # Specifying the number of epochs for training
    num_epochs = 200

    # Training the model
    loss_hist = train_model(encoder,decoder,device,loss_fn,optim,xArr,num_epochs)

    # Passing the data through the encoder and the decoder. z's
    # are the latent vector presentations and xHat is the
    # reconstructed signal.
    [z,xHat] = eval_encdec(encoder,decoder,device,xArr)
    return loss_hist, xArr, xHat, yArr, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We do some visualization of the latent space using t-SNE and use the reconstruction error as an indicator of anomaly. The ROC curve shows the trade-off between false and true positives as the threshold for reconstruction error changes.
    """)
    return


@app.cell
def _(decoder, device, encoder, plot_ROC, plt, tsne_plot, xArr, yArr, z):
    _f = plt.figure()
    _f.set_figwidth(12)
    _f.set_figheight(5)

    # Doing a scatter plot visualization of the raw data
    plt.subplot(1,2,1)
    tsne_plot(z,yArr)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('t-SNE Visualization')

    # Plotting the ROC curve when using recnstruction error for classification
    plt.subplot(1,2,2)
    auc = plot_ROC(encoder,decoder,device,xArr,yArr)
    plt.grid()
    plt.title("ROC Curve for Anomaly Det | AUC = {:1.3f}".format(auc))

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following sliders allow you to select a window that is normal vs. one that is abnormal for visual inspection of their reconstruction.
    """)
    return


@app.cell
def _(mo, np, xArr, yArr):
    # Counts the no. of normal windows, i.e., when yArr = False
    _nNormal = np.sum(yArr == False)

    # Slider for selecting the Record ID for the data
    win_id_normal_slider = mo.ui.slider(0, _nNormal-1, step=1, value=0, label="Normal Win ID", debounce=True)
    win_id_abnormal_slider = mo.ui.slider(0, len(xArr)-1-_nNormal, step=1, value=0, label="Anomaly Win ID", debounce=True)

    mo.hstack([win_id_normal_slider, win_id_abnormal_slider])
    return win_id_abnormal_slider, win_id_normal_slider


@app.cell
def _(
    loss_hist,
    np,
    plt,
    win_id_abnormal_slider,
    win_id_normal_slider,
    xArr,
    xHat,
    yArr,
):
    _f = plt.figure()
    _f.set_figwidth(18)
    _f.set_figheight(5)

    # Plotting the learning curves
    plt.subplot(1,3,1)
    plt.plot(loss_hist)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.grid()

    # Indices of the normal and abnormal windows
    _normalIdx = np.where(yArr == False)[0]
    _abnormalIdx = np.where(yArr == True)[0]

    # Plotting a normal observation and its reconstruction
    plt.subplot(1,3,2)
    _idx = _normalIdx[win_id_normal_slider.value]
    plt.plot(xArr[_idx,:])
    plt.plot(xHat[_idx,:])
    plt.legend(['Observation', 'Prediction'])
    _rmse = np.sqrt(np.mean((xArr[_idx,:] - xHat[_idx,:]) ** 2))
    plt.title(f'Normal (RMSE = {_rmse:.4f})')
    plt.grid()

    # Plotting an abnormal observation and its reconstruction
    plt.subplot(1,3,3)
    _idx = _abnormalIdx[win_id_abnormal_slider.value]
    plt.plot(xArr[_idx,:])
    plt.plot(xHat[_idx,:])
    plt.legend(['Observation', 'Prediction'])
    _rmse = np.sqrt(np.mean((xArr[_idx,:] - xHat[_idx,:]) ** 2))
    plt.title(f'Abnormal (RMSE = {_rmse:.4f})')
    plt.grid()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
