import torch
from data.dataloader import data_loader_train
from models.Networks import Networks
import models.metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize


def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

num = 2500
k_num = 5
Pred = []

data_0 = sio.loadmat('MNIST/BDGP91.mat')
data_dict = dict(data_0)
data0 = data_dict['groundtruth'].T
label_true = np.zeros(2500)
for i in range(2500):
    label_true[i] = data0[i]
print(label_true)

reg2 = 1.0 * 10 ** (k_num / 10.0 - 3.0)

model = Networks()
#model.load_state_dict(torch.load('./models/AE1.pth'))

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.0)
n_epochs = 50
for epoch in range(n_epochs):
    for data in data_loader_train:
        train_imga, train_imgb = data
        input1 = train_imga
        input2 = train_imgb
        output1, output2 = model(input1, input2)
        loss = 0.5*criterion(output1, input1) + 0.5*criterion(output2, input2)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    if epoch % 1 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("Loss is:{:.4f}".format(loss.item()))
torch.save(model.state_dict(), './models/AE1.pth')

print("step2")
print("---------------------------------------")
criterion2 = torch.nn.MSELoss(reduction='mean')
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.0)
n_epochs2 = 150
ACC = []
NMI = []
PUR = []

for epoch in range(n_epochs2):
    for data in data_loader_train:
        train_imga, train_imgb = data
        input1 = train_imga
        input2 = train_imgb

        output11, output22, out_1, out_2, out11_, out12_, coef, out11, out22 = model.forward2(input1, input2)

        coef_12 = torch.Tensor(coef)
        coef = torch.norm(coef_12, p=1, dim=0)
        loss_selfc12 = torch.norm(coef, p=2)

        loss_selfc = criterion2(coef, torch.zeros(2500, 2500, requires_grad=True))

        loss_cca = model.loss(out11_, out12_)

        loss_self = 0.5*criterion2(out_1, out11) + 0.5*criterion2(out_2, out22)

        loss_d = 0.5 * criterion2(output11, input1) + 0.5 * criterion2(output22, input2)

        loss = 1*loss_d + 1*loss_self + 1*loss_selfc + loss_cca + loss_selfc12

        optimizer2.zero_grad()
        loss.backward(retain_graph=True)
        optimizer2.step()
    if epoch % 1 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs2))
        print("Loss is:{:.4f}".format(loss.item()))
        print("Loss_cca is:{:.4f}".format(loss_cca.item()))
        print("loss_selfc is:{:.4f}".format(loss_selfc.item()))
        print("loss_selfc12 is:{:.4f}".format(loss_selfc12.item()))
        print("loss_self is:{:.4f}".format(loss_self.item()))
        print("loss_d is:{:.4f}".format(loss_d.item()))

        coef = model.weight - torch.diag(torch.diag(model.weight))
        commonC = coef.cpu().detach().numpy()
        alpha = max(0.4 - (k_num - 1) / 10 * 0.1, 0.1)
        All_C = thrC(commonC, alpha)
        preds, _ = post_proC(All_C, k_num)

        acc = metrics.acc(label_true, preds)
        nmi = metrics.nmi(label_true, preds)
        pur = metrics.purity_score(label_true, preds)
        ACC.append(acc)
        NMI.append(nmi)
        PUR.append(pur)
        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f， pur: %.4f  <==|'
              % (acc, nmi, pur))

        if acc > 0.955:
            C_path = 'All_C' + str(epoch)
            sio.savemat(C_path + '.mat', {'C': All_C})
            sio.savemat('Pred' + str(epoch) + '.mat', {'Pred': preds})

sio.savemat('ACC' + '.mat', {'acc': ACC})
sio.savemat('NMI' + '.mat', {'nmi': NMI})
sio.savemat('PUR' + '.mat', {'pur': PUR})

torch.save(model.state_dict(), './models/AE2.pth')




