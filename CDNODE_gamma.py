import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as io
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--visualize', type=eval, default=False)
#parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--rtol', type=float, default=1e-7)
parser.add_argument('--atol', type=float, default=1e-13)
parser.add_argument('--PATIENCE', type=int, default=30)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--nhidden', type=int, default=64)
parser.add_argument('--kk',type=int,default=10)
parser.add_argument('--seq', type=int, default=100) #200
parser.add_argument('--scale',type=float,default = 10)
parser.add_argument('--scale2',type=float,default = 0.15)
parser.add_argument('--seed',type=int,default = 0)
parser.add_argument('--dim',type=int,default = 0)
parser.add_argument('--savedir',type=str,default = None)
parser.add_argument('--delay',type=int,default = 50)
parser.add_argument('--size',type=int,default = 9) #training set size
parser.add_argument('--fold',type=int,default = 1) #k fold

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def chunking_frame(data, window, overlap):
    shift = window - overlap
    frame_num = int(np.ceil(data.shape[0] / float(shift)))
    chunked_data = np.zeros((frame_num, window, data.shape[1]), dtype='float')
    i_w = 0
    for wn in range(frame_num):
        if (wn == frame_num - 1):
            chunked_data[wn, 0:np.shape(data[i_w:, :])[0], :] = data[i_w:, :]
        else:
            chunked_data[wn, 0:window, :] = data[i_w:i_w + window, :]
        i_w = i_w + shift
    return chunked_data


def Binomialfilter(delay_frame):
    h = [0.5, 0.5]
    j = int(2 * delay_frame + 1)
    binomialcoff = np.convolve(h, h)
    for i in range(j - 3):
        binomialcoff = np.convolve(binomialcoff, h)
    binomialcoff = binomialcoff / (np.sum(binomialcoff))
    return binomialcoff

def CCCloss(pred_y, true_y):
    ux = torch.mean(pred_y)
    uy = torch.mean(true_y)
    pred_y1 = pred_y - ux
    true_y1 = true_y - uy
    # pred_y1 = pred_y1[0:len(true_y1)]
    cc = torch.sum(pred_y1 * true_y1) / (
            torch.sqrt(torch.sum(pred_y1 ** 2)) * torch.sqrt(torch.sum(true_y1 ** 2)))
    # print(cc.device)

    if device.type == 'cpu':
        cc_num = torch.tensor(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device.type)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.tensor(1.0) - ccc
    else:
        cc_num = torch.cuda.FloatTensor(1, ).fill_(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.cuda.FloatTensor(1, ).fill_(1.0) - ccc
    return ccc


def cc_cal(pred_y, true_y):
    ux = torch.mean(pred_y)
    uy = torch.mean(true_y)
    pred_y1 = pred_y - ux
    true_y1 = true_y - uy
    # pred_y1 = pred_y1[0:len(true_y1)]
    cc = torch.sum(pred_y1 * true_y1) / (
            torch.sqrt(torch.sum(pred_y1 ** 2)) * torch.sqrt(torch.sum(true_y1 ** 2)))

    cc_num = 2.0 * cc * (torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2)))

    ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))

    cc = cc.cpu().detach().numpy()
    ccc = ccc.cpu().detach().numpy()
    return ccc

class PJClassifier(nn.Module):# projection to new space

    def __init__(self, inp_dim, latent_dim, use_gpu):
        super(PJClassifier, self).__init__()

        self.layer1 = nn.Linear(inp_dim, latent_dim)
        #self.layer2 = nn.Linear(latent_dim, 1)

        self.use_gpu = use_gpu
        self.latent_dim = latent_dim

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out) # this out is the embeddings
        #ini= self.layer2(out[0,:]) # this out is initial values
        return out

class initpred(nn.Module):# projection to new space

    def __init__(self, latent_dim, out_dim, use_gpu):
        super(initpred, self).__init__()

        self.layer1 = nn.Linear(latent_dim, out_dim)
        self.use_gpu = use_gpu

    def forward(self, x):
        out = self.layer1(x)
       # out = torch.tanh(out) # this out is the initial values
        out = torch.sigmoid(out)
        return out


class LatentODEfunc(nn.Module):

    def __init__(self, inp_dim, nhidden, out_dim):
        super(LatentODEfunc, self).__init__()
        #self.elu = nn.tanh(inplace=True)
        self.fc1 = nn.Linear(inp_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, out_dim)
        # self.alpha = nn.Linear(inp_dim, out_dim)
        # self.alpha = nn.Sequential( 
        #    nn.Linear(inp_dim, nhidden), 
        #    nn.Linear(nhidden, out_dim), 
        #    # nn.ReLU(),
        #    # nn.Dropout(dropout)
        # )
        self.nfe = 0
        

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        
        # alpha = self.alpha(x)
        
        if args.scale != 0:
            out = args.scale * torch.tanh(out/args.scale)
        
        # out = alpha * (torch.tanh((1/alpha) *out)) #alphs = 1??
        # out = self.alpha(x) * out
        #out = (torch.tanh(out)+1.0)/2.0*2.5-0.9583
        #out = (torch.tanh(out) + 1.0) / 2.0 * 4 - 2
        return out

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


################# Chunking data ########################################################################################
args.seed = 10
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# dim = 0
dim = args.dim
# delay = 50
delay = args.delay
ts_size = 7501
ts_size2 = 7501
batch_size = int(ts_size) - int(delay)

out_dim = 1 #equal to the output dimension
latent_dim = args.latent_dim
nhidden = args.nhidden
bcoef = Binomialfilter(delay)


name = [str(nhidden) + 'artanh'+str(args.scale),str(nhidden) + 'vrtanh'+str(args.scale)]

ftype = 2
features = ['mfcc', 'eGemaps', 'boaw']



#filepath = "/home/561/jw3506/CNODE/CD-NODEs/varying_train_size/size_"+ str(args.size) + "_"+str(args.fold) + "/"
#savepath = "/home/561/jw3506/CNODE/emo_out_good_new_alpha/"

filepath = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/CD-NODEs/"
#savepath = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/emo_out_good2/"
modelpath = '/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/emo_out_good/.pth'

filepath2 = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/Interspeech2023/CODE/"

beta_path = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/Interspeech2023/CODE/beta_estimate_jen/"
savepath = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/CD-NODEs/examples/interspeech2023/good_new/results/"
savepath = beta_path
model = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/CD-NODEs/examples/interspeech2023/good_new/results/MSEloss2/"

#savepath = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/CD-NODEs/examples/interspeech2023/final_results/SD/"
#savepath = '/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/CD-NODEs/examples/interspeech2023/good_new/MAP_results/SD/'
# modelpath = 'C:/Users/EET/OneDrive - UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/emo_out_good/.pth'
# modelpath2 = 'C:/Users/EET/OneDrive - UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/emo_out_good/pri.pth'
# modelpath3 = 'C:/Users/EET/OneDrive - UNSW/0UNSW_PhD/ICASSP2023/CD-NODEs_new/emo_out_good/rec.pth'
savepath = "/Users/jingyaowu/Library/CloudStorage/OneDrive-UNSW/0UNSW_PhD/Interspeech 2024/MAP_estimation/cdnode_results/new/"

import os
# Check whether the specified path exists or not
isExist = os.path.exists(savepath)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(savepath)
   print("The new directory is created!")


if __name__ == '__main__':
    
    #path = savepath +'sizez_'+str(args.size)+'fold'+str(args.fold) + '/seed'+ str(args.seed) + '/alpha' + str(args.scale)

   # if not os.path.exists(path):
   #     os.makedirs(path)
    
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        ugpu = 0
    else:
        ugpu = 1

    seq = args.seq
    overlap = 0

    ################# lOAD DATA ############################################################################################
    
    if args.delay == 50:
        print("Loading festures..")
        data_all = io.loadmat(filepath + 'boaw_2s_cut.mat')
        X_train = data_all['data_train']
        X_test = data_all['data_dev']
    
        ################# lOAD DATA ############################################################################################
       # print("Loading labels...")
       # data_all = io.loadmat(filepath + 'goldstand_2s_cut.mat')
       # Y_train = data_all['gt_train']
       # Y_test = data_all['gt_dev']
        
        print("Loading sigma...")
        data_all = io.loadmat(beta_path + 'MLE_beta_ar_mu_sd_2s.mat')
        #data_all = io.loadmat(beta_path + 'MAP_beta_estimate_mu_sd_good.mat')
        
        Y_train = data_all['ar_train_sd']
        Y_test = data_all['ar_test_sd']
        
        print("Loading mu...")
        data_all = io.loadmat(beta_path + 'MLE_beta_ar_mu_sd_2s.mat')
        #data_all = io.loadmat(beta_path + 'MAP_beta_estimate_mu_sd_good.mat')
        
        Y_train2 = data_all['ar_train_mu']
        Y_test2 = data_all['ar_test_mu']
    else:
        data_all = io.loadmat(filepath + 'boaw_4s_cut.mat')
        X_train = data_all['data_train']
        X_test = data_all['data_dev']
    
        ################# lOAD DATA ############################################################################################
        # print("Loading labels...")
        # data_all = io.loadmat(filepath + 'goldstand_4s_cut.mat')
        # Y_train = data_all['gt_train']
        # Y_test = data_all['gt_dev']
        
        # data_all = io.loadmat(beta_path + 'MLE_beta_ar_mu_sd_4s.mat')
        # #Y_train = data_all['gt_train']
        # #Y_test = data_all['gt_dev']
        # Y_train = data_all['ar_train_mu']
        # Y_test = data_all['ar_test_mu']
        
        print("Loading sigma...")
        data_all = io.loadmat(beta_path + 'MLE_beta_ar_mu_sd_4s_new.mat')
        Y_train = data_all['ar_train_sd']
        Y_test = data_all['ar_test_sd']

    Y_train = Y_train[:,dim]
    Y_test = Y_test[:,dim]
    
    Y_train2 = Y_train2[:,dim]
    Y_test2 = Y_test2[:,dim]

    inp_dim = X_train.shape[1]  # input feature dimension


    ########## Initilizing matrix and parameters for network  #####################
    ccc_train = np.zeros((args.nepochs + 1,))
    ccc_test = np.zeros((args.nepochs + 1,))
    ccc_train_ave = np.zeros((args.nepochs + 1,))
    ccc_test_ave = np.zeros((args.nepochs + 1,))
    
    ccc_best = -1000 * np.ones(1, )
    train_loss = np.zeros((args.nepochs + 1,))
    
    ccc_train2 = np.zeros((args.nepochs + 1,))
    ccc_test2 = np.zeros((args.nepochs + 1,))
    ccc_train_ave2 = np.zeros((args.nepochs + 1,))
    ccc_test_ave2 = np.zeros((args.nepochs + 1,))
    
    ccc_best2 = -1000 * np.ones(1, )
    train_loss2 = np.zeros((args.nepochs + 1,))

    ############## Forming pytorch dataset ###############
    X_train = torch.from_numpy(X_train).float() # train data
    Y_train = torch.from_numpy(Y_train).float() # train labels
    
    Y_train2 = torch.from_numpy(Y_train2).float() # train labels mu
    #reduce training set size
    # reduced_size = 7501*7
    # X_train = X_train[0:reduced_size,:]
    # Y_train = Y_train[0:reduced_size]


    X_test = torch.from_numpy(X_test).float()  # train data
    Y_test = torch.from_numpy(Y_test).float()  # train labels

    Y_test2 = torch.from_numpy(Y_test2).float()  # train labels mu
    
    # train_dataset = TensorDataset(X_train,Y_train)
    # test_dataset = TensorDataset(X_test, Y_test)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset,
    #                           batch_size=ts_size*9,
    #                           shuffle=False)
    # batches_per_epoch = len(train_loader)
    
    
    train_dataset = TensorDataset(X_train,Y_train,Y_train2)
    test_dataset = TensorDataset(X_test, Y_test,Y_test2)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=ts_size*9,
                              shuffle=False)
    batches_per_epoch = len(train_loader)

    #### model
    rec = PJClassifier(inp_dim, latent_dim, ugpu).to(device)
    rec2 = PJClassifier(inp_dim, latent_dim, ugpu).to(device)
    # rec.apply(init_weights)
    # rec.load_state_dict(torch.load(modelpath3))
    
    pri = initpred(latent_dim,out_dim,ugpu).to(device)
    pri2 = initpred(latent_dim,out_dim,ugpu).to(device)
    # pri.load_state_dict(torch.load(modelpath2))
    
    func = LatentODEfunc(latent_dim + out_dim, nhidden, out_dim).to(device)
    func2 = LatentODEfunc(latent_dim + out_dim, nhidden, out_dim).to(device)
    # func.apply(init_weights)
    # func.load_state_dict(torch.load(modelpath)) #load the saved initilizations
    # random_seed = 1
    # torch.manual_seed(random_seed)
    # torch.nn.init.xavier_normal_(func.fc1.weight)
    # torch.nn.init.xavier_normal_(func.fc2.weight)
    # torch.nn.init.xavier_normal_(func.fc3.weight)
    
    
    
   
    # func.eval()
    #params = (list(rec.parameters()) + list(func.parameters()) + list(pri.parameters()))
    params = (list(rec.parameters()) + list(func.parameters()) + list(pri.parameters()) + list(rec2.parameters()) + list(func2.parameters()) + list(pri2.parameters())   )
    #params2 = (list(rec2.parameters()) + list(func2.parameters()) + list(pri2.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    #optimizer2 = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    #scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer=optimizer2, gamma=0.9)

#save initialization 
# Initialize model
    # model = TheModelClass()
    
    # # Initialize optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in func.state_dict():
    #     print(param_tensor, "\t", func.state_dict()[param_tensor].size())
    
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])
        
    modelpath3 = model + 'rec' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) + '_' + name[dim] +'_delay' + str(args.delay) + '.pth'
    modelpath_pri =  model + 'pri' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) + '_' + name[dim] +'_delay' + str(args.delay)+ '.pth'
    modelpath_func =  model + 'func' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) + '_' + name[dim] +'_delay' + str(args.delay)+ '.pth'

    
    # rec = PJClassifier(inp_dim, latent_dim, ugpu).to(device)
    # torch.save(rec.state_dict(), modelpath3)
    # # func = LatentODEfunc(latent_dim + out_dim, nhidden, out_dim).to(device)
    # # pri = initpred(latent_dim,out_dim,ugpu).to(device)
    
    
    # rec = torch.load(modelpath3)
    # rec.eval()
    # pri = torch.load(modelpath_pri)
    # pri.eval()
    # func = torch.load(modelpath_func)
    # func.eval()

    # rec.load_state_dict(torch.load(modelpath3))
    # rec.eval()
    # pri.load_state_dict(torch.load(modelpath_pri))
    # pri.eval()
    # func.load_state_dict(torch.load(modelpath_func))
    # func.eval()

    loss_meter = RunningAverageMeter()
    loss_meter2 = RunningAverageMeter()

    for itr in range(args.nepochs+1): # epoch number
        loss_save = np.zeros((len(train_loader) + 1,))
        ccc_save = np.zeros((len(train_loader) + 1,))
        for i,data in enumerate(train_loader):
            print('Epoch = ' + str(itr) + ' Iteration ='+ str(i))
            inputs,labels,mu = data

            inputs = rec.forward(inputs.float().to(device))  # converting to low dimensions of all features

            inputs = torch.from_numpy(chunking_frame(inputs.cpu().detach().numpy(), seq, overlap)).permute(1,0,2)
            labels_new = torch.from_numpy(chunking_frame(torch.unsqueeze(labels,1).cpu().detach().numpy(), seq, overlap)).permute(1,0,2)
            mu_new = torch.from_numpy(chunking_frame(torch.unsqueeze(mu,1).cpu().detach().numpy(), seq, overlap)).permute(1,0,2)
            

            inputs = inputs.float().to(device)  # train data
            labels = labels.float().to(device)  # train labels
            mu = mu.float().to(device)  # train labels
            print(inputs.shape)
            optimizer.zero_grad()
            #optimizer2.zero_grad()
            ############################ Start trainiing the model #######
           # s0 = labels_new[0,:,:].float().to(device)
          #  C0 = 1
            #y0_new_sd = torch.atanh(2 * s0 / C0 - 1)
            
            
            ts = torch.linspace(0, (seq-1)*0.04, steps=seq)
            #y0_new = pri.forward(inputs[0,:,:]).float().to(device) #predict all initial values of all the chunks
            y0_new = labels_new[0,:,:].float().to(device)
            y0_new2 = mu_new[0,:,:].float().to(device)
            #y0_new = torch.atanh(2 * y0_new - 1) 
            pred_train = odeint(func, y0_new, ts, inputs, args.rtol,args.atol)
            pred_train_mu = odeint(func2, y0_new2, ts, inputs, args.rtol,args.atol)

            #pred_train = 0.5 * torch.tanh(pred_train) + 0.5 # add constraint
            pred_train = args.scale2 *torch.sigmoid(pred_train)
            gmma = 0.75
            pred_train_mu = gmma*torch.sigmoid(pred_train_mu)
            ################## plot #####################
            pred_train = torch.reshape(torch.t(pred_train[:, :, 0]), (-1,))
            pred_train_mu = torch.reshape(torch.t(pred_train_mu[:, :, 0]), (-1,))
            
            
            #remove nan
            if torch.isnan(pred_train).any():
                id = torch.nonzero(torch.isnan(pred_train))
                pred_train[id] = pred_train[id-1]
                print("Ops, NaN!")
                
            print(pred_train.shape)
            print(labels.shape)
            eta = 15
            loss = eta*CCCloss(pred_train[:batch_size],labels)+CCCloss(pred_train_mu[:batch_size],mu)
            
            #loss_func = nn.MSELoss()
            #loss = loss_func(pred_train[:batch_size],labels)
            loss_save[i] = loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            ccc_save[i] = cc_cal(pred_train[:batch_size],labels)
            #ccc_save2[i] = cc_cal(pred_train_mu[:batch_size],mu)
            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))



            #################################################################
            if i == 0:
                p_ts = pred_train[:batch_size]
                p_ts_mu = pred_train_mu[:batch_size]
            else:
                p_ts = torch.cat((p_ts, pred_train[:batch_size]))
                p_ts_mu = torch.cat((p_ts_mu, pred_train_mu[:batch_size]))
        
        train_loss[itr] = np.mean(loss_save)
        ccc_train[itr] = cc_cal(p_ts, Y_train.to(device))
        ccc_train_ave[itr] = np.mean(ccc_save)
        print('SD Train CCC is of ' + str(itr) + ' is ' + str(ccc_train[itr]))
        print('Average Train CCC is of ' + str(itr) + ' is ' + str(ccc_train_ave[itr]))
        
        ccc_train2[itr] = cc_cal(p_ts_mu, Y_train2.to(device))
        #ccc_train_ave2[itr] = np.mean(ccc_save)
        print('MU Train CCC is of ' + str(itr) + ' is ' + str(ccc_train2[itr]))
        #print('Average Train CCC is of ' + str(itr) + ' is ' + str(ccc_train_ave[itr]))

        scheduler.step()
       # scheduler2.step()
        
        plt.plot(pred_train.detach().numpy(), label = 'pred_sd')
        plt.plot(labels.detach().numpy(),label = 'gt sd')
        plt.legend(loc="lower right")
        plt.show()
        
        plt.plot(pred_train_mu.detach().numpy(), label = 'pred_MU')
        plt.plot(mu.detach().numpy(),label = 'gt MU')
        plt.legend(loc="lower right")
        plt.show()

        with torch.no_grad(): ### after each epoch training
            ######################## Test may be able to be changed
            ccc_save_test = np.zeros((len(test_loader) + 1,))
            for i, data in enumerate(test_loader):# for each utterence
                te_fea, te_lab, MUt = data
                te_fea = rec.forward(te_fea.float().to(device))  # converting to low dimensions of all features

                te_fea=chunking_frame(te_fea.cpu().detach().numpy(), ts_size2, 0)
                te_fea=torch.from_numpy(te_fea).float().to(device)
                te_fea = te_fea.permute(1, 0, 2)

                te_lab1 = chunking_frame(torch.unsqueeze(te_lab, 1).cpu().detach().numpy(), ts_size2, 0)
                te_lab1 = torch.from_numpy(te_lab1).float().to(device)  # train labels
                te_lab1 = te_lab1.permute(1, 0, 2)
                tt = torch.linspace(0, (te_fea.shape[0] - 1) * 0.04, steps=te_fea.shape[0])
                
                te_lab1_mu = chunking_frame(torch.unsqueeze(MUt, 1).cpu().detach().numpy(), ts_size2, 0)
                te_lab1_mu = torch.from_numpy(te_lab1_mu).float().to(device)  # train labels
                te_lab1_mu = te_lab1_mu.permute(1, 0, 2)
                tt_mu = torch.linspace(0, (te_fea.shape[0] - 1) * 0.04, steps=te_fea.shape[0])
                
                #y0_new2 = torch.atanh(2 * te_lab1[delay,:,:] - 1) 
                #p_test = odeint(func, y0_new2, tt, te_fea, args.rtol, args.atol) #delay compensation
                #y0_pri = pri.forward(te_fea[0,:,:]).float().to(device)
                y0_pri = te_lab1[delay,:,:]
                p_test = odeint(func, y0_pri, tt, te_fea, args.rtol, args.atol) #delay compensation
                # p_test = odeint(func, te_lab1, tt, te_fea, args.rtol, args.atol)
                #p_test = 0.5 * torch.tanh(p_test) + 0.5
                p_test = args.scale2*torch.sigmoid(p_test)
                
                y0_pri_mu = te_lab1_mu[delay,:,:]
                p_test_mu = odeint(func2, y0_pri_mu, tt_mu, te_fea, args.rtol, args.atol) #delay compensation
                # p_test = odeint(func, te_lab1, tt, te_fea, args.rtol, args.atol)
                #p_test = 0.5 * torch.tanh(p_test) + 0.5
                
                p_test_mu = gmma*torch.sigmoid(p_test_mu)
                
                ################## post processing for delay per utterence ##########
                p_test = p_test[:ts_size2,:,:]
                p_temp = p_test[0,:,:].repeat(int(delay),1,1)
                # p_test = torch.cat((p_temp,p_test[:-int(delay) + 1,:,:]),0)
                p_test = torch.cat((p_temp,p_test[:-int(delay),:,:]),0)
                
                p_test_mu = p_test_mu[:ts_size2,:,:]
                p_temp_mu = p_test_mu[0,:,:].repeat(int(delay),1,1)
                # p_test = torch.cat((p_temp,p_test[:-int(delay) + 1,:,:]),0)
                p_test_mu = torch.cat((p_temp_mu,p_test_mu[:-int(delay),:,:]),0)
                
                #remove nan
                if torch.isnan(p_test).any():
                    id = torch.nonzero(torch.isnan(p_test))            
                    p_test[id] = p_test[id-1]
                    print("Ops, NaN!")
                    
                ccc_save_test[i] = cc_cal(p_test,te_lab1)
                
                #################################################################
                if i == 0:
                    p_ts = torch.reshape(torch.t(p_test[:ts_size2,:,0]),(-1,))
                    p_ts_mu = torch.reshape(torch.t(p_test_mu[:ts_size2,:,0]),(-1,))
                else:
                    p_ts = torch.cat((p_ts, torch.reshape(torch.t(p_test[:ts_size2,:,0]),(-1,))))
                    p_ts_mu = torch.cat((p_ts_mu, torch.reshape(torch.t(p_test_mu[:ts_size2,:,0]),(-1,))))

            ccc_test[itr] = cc_cal(p_ts,Y_test.to(device))
            ccc_test_ave[itr] = np.mean(ccc_save_test)
            print('SD Test CCC is of ' + str(itr) + ' is ' + str(ccc_test[itr]))
            print('Average Test CCC is of ' + str(itr) + ' is ' + str(ccc_test_ave[itr]))
            
            ccc_test2[itr] = cc_cal(p_ts_mu,Y_test2.to(device))
            #ccc_test_ave[itr] = np.mean(ccc_save_test)
            print('MU Test CCC is of ' + str(itr) + ' is ' + str(ccc_test2[itr]))
            #print('Average Test CCC is of ' + str(itr) + ' is ' + str(ccc_test_ave[itr]))

            if ccc_test[itr] > ccc_best[0]:
                ccc_best[0] = ccc_test[itr]
                curr_step = 0
                print('CCC best = ' + str(ccc_best[0]) + ' epoch = ' + str(itr))
                io.savemat(
                    savepath + 'SD_p_tsfd_' + name[dim] +'_delay' + str(args.delay) +'_alpha' + str(args.scale) +'_' + str(args.scale2) + '_' + str(gmma) +'_loss'+str(eta) + '.mat',
                    {"p_test": p_ts.cpu().detach().numpy()})
                io.savemat(
                    savepath + 'MU_p_tsfd_' + name[dim] +'_delay' + str(args.delay) +'_alpha' + str(args.scale) +'_' + str(gmma) +'_loss'+str(eta) +'.mat',
                    {"p_test": p_ts_mu.cpu().detach().numpy()})
                #io.savemat(savepath + '/p_tsfd_' + '_' + name[dim] + '.mat', {"p_test": p_ts.cpu().detach().numpy()})
                # io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) + '_CCC_tests_' + '_' + name[dim] + '.mat',{"ccc_test": ccc_test})
                # io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_CCC_trains_' + '_' + name[dim] + '.mat',{"ccc_train": ccc_train})
                # io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) + '_train_loss' + '_' + name[dim] + '.mat', {"train_loss": train_loss})
             
            plt.plot(p_ts[:batch_size].detach().numpy(), label = 'test pred_sd')
            plt.plot(te_lab[:batch_size].detach().numpy(),label = 'test gt sd')
            plt.legend(loc="lower right")
            plt.show()
            
            plt.plot(p_ts_mu[:batch_size].detach().numpy(), label = 'test pred_MU')
            plt.plot(MUt[:batch_size].detach().numpy(),label = 'test gt MU')
            plt.legend(loc="lower right")
            plt.show()
                  #  else:
                #curr_step += 1
                # if curr_step == args.PATIENCE:
                    # print('Early Sopp!')
                    # break
    io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) + '_CCC_tests_' + '_' + name[dim] +'_delay' + str(args.delay)+'_loss'+str(eta) + '.mat',{"ccc_test": ccc_test})
    io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) +'_CCC_trains_' + '_' + name[dim] +'_delay' + str(args.delay) + '_loss'+str(eta) + '.mat',{"ccc_train": ccc_train})
    io.savemat(savepath + 'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) + '_train_loss' + '_' + name[dim] +'_delay' + str(args.delay) + '_loss'+str(eta) + '.mat', {"train_loss": train_loss})
    torch.save(rec.state_dict(),  savepath + 'rec' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) + '_' + name[dim] +'_delay' + str(args.delay)+'_loss'+str(eta)  + '.pth')
    torch.save(pri.state_dict(), savepath + 'pri' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) + '_' + name[dim] +'_delay' + str(args.delay)+'_loss'+str(eta) + '.pth')
    torch.save(func.state_dict(), savepath + 'func' +'SD_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(args.scale2) + '_' + name[dim] +'_delay' + str(args.delay)+'_loss'+str(eta) + '.pth')

  #  io.savemat(
   #     savepath + 'MU_p_tsfd_' + name[dim] +'_delay' + str(args.delay) +'_alpha' + str(args.scale) +'_' + str(0.8) + '.mat',
    #    {"p_test": p_ts_mu.cpu().detach().numpy()})
    io.savemat(savepath + 'MU_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(gmma) + '_CCC_tests_' + '_' + name[dim] +'_delay' + str(args.delay) + '_loss'+str(eta) +'.mat',{"ccc_test": ccc_test2})
    io.savemat(savepath + 'Mu_seed'+ str(args.seed) + '_alpha' + str(args.scale) +'_' + str(gmma) +'_CCC_trains_' + '_' + name[dim] +'_delay' + str(args.delay) + '_loss'+str(eta) +'.mat',{"ccc_train": ccc_train2})

    # io.savemat(savepath + 'seed'+ str(args.seed) + '/alpha' + str(args.scale) + '/CCC_tests_' + '_' + name[dim] + '.mat',{"ccc_test": ccc_test})
    # io.savemat(savepath + 'seed'+ str(args.seed) + '/alpha' + str(args.scale) +'/CCC_trains_' + '_' + name[dim] + '.mat',{"ccc_train": ccc_train})
    # io.savemat(savepath + 'seed'+ str(args.seed) + '/alpha' + str(args.scale) + '/train_loss' + '_' + name[dim] + '.mat', {"train_loss": train_loss})

  #  io.savemat(path + '/CCC_tests_' + '_' + name[dim] + '.mat',{"ccc_test": ccc_test})
  #  io.savemat(path +'/CCC_trains_' + '_' + name[dim] + '.mat',{"ccc_train": ccc_train})
  #  io.savemat(path + '/train_loss' + '_' + name[dim] + '.mat', {"train_loss": train_loss})
  #  io.savemat(path + '/CCC_tests_ave_' + '_' + name[dim] + '.mat',{"ccc_test_ave": ccc_test_ave})
  #  io.savemat(path +'/CCC_trains_ave_' + '_' + name[dim] + '.mat',{"ccc_train_ave": ccc_train_ave})



    # io.savemat(savepath + args.savedir +'/CCC_tests_' + str(args.scale) + name[dim] + features[ftype] + '.mat',{"ccc_test": ccc_test})
    # io.savemat(savepath + args.savedir + '/CCC_trains_' + str(args.scale) + name[dim] + features[ftype] + '.mat',{"ccc_train": ccc_train})
    # io.savemat(savepath + args.savedir + '/train_loss'+ str(args.scale) + name[dim] + features[ftype] + '.mat', {"train_loss": train_loss})
    # io.savemat(savepath + + args.savedir + 'p_tsfd_' + str(args.scale) + name[dim] + features[ftype] + '.mat',
    #     {"p_test": p_ts.cpu().detach().numpy()})
    del train_dataset
    del test_dataset
    del train_loader
    del test_loader
    del rec, pri, func, params
    torch.cuda.empty_cache()
