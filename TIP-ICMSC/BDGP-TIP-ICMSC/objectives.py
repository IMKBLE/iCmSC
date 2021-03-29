import torch
import scipy.io as sio


class cca_loss():
    def __init__(self, outdim_size):
        self.outdim_size = outdim_size
        #self.device = device
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t() #转置
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0
        # sio.savemat('hello.mat', {'Z': H1.cpu().detach().numpy()})

        o1 = o2 = H1.size(0) #o1结果是10

        m = H1.size(1) #m是4200
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        #print(H1.shape)#[10,4200]
        #print(H1.mean(dim=1).unsqueeze(dim=1).shape)#[10,1]
        #print(H1bar.shape)#[10,4200]
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)#[10,4200]
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0


        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1)#[10,10]
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)#D1大小[10], V1大小[10,10]
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        #print(D1.shape)
        #print(V1.shape)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        #print(posInd1.size())
        #print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        # just the top self.outdim_size singular values are used
        U, V = torch.symeig(torch.matmul(
            Tval.t(), Tval), eigenvectors=True)
        # U = U[torch.gt(U, eps).nonzero()[:, 0]]
        # print(U.shape)
        U = U.topk(self.outdim_size)[0]
        # sio.savemat('hello.mat', {'Z': U.cpu().detach().numpy()})
        corr = torch.sum(torch.sqrt(U))
        return -corr
