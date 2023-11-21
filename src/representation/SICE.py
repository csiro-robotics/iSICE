import profile
import torch
import torch.nn as nn
from torch.autograd import Function
from numpy import linalg as LA
import numpy as np

class SICE(nn.Module):
     def __init__(self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=None, sparsity_val=0.0, sice_lrate=0.0):

         super(SICE, self).__init__()
         self.iterNum=iterNum
         self.is_sqrt = is_sqrt
         self.is_vec = is_vec
         self.dr = dimension_reduction
         self.sparsity = sparsity_val
         self.learingRate = sice_lrate
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
               nn.BatchNorm2d(self.dr),
               nn.ReLU(inplace=True)
             )
         output_dim = self.dr if self.dr else input_dim
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()

     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def _cov_pool(self, x):
         return Covpool.apply(x)
     

     def _inv_sqrtm(self, x, iterN):
         return Sqrtm.apply(x, iterN)

     def _sqrtm(self, x, iterN):
         batchSize = x.shape[0]
         dim = x.shape[1]
         dtype = x.dtype
         I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
         normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
         ZY = 0.5 * (I3 - A)
         if iterN < 2:
             ZY = 0.5*(I3 - A)
             YZY = A.bmm(ZY)
         else:
             Y = A.bmm(ZY)
             Z = ZY
             for _ in range(iterN - 2):
                 ZY = 0.5 * (I3 - Z.bmm(Y))
                 Y = Y.bmm(ZY)
                 Z = ZY.bmm(Z)
             YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
         y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         return y
     
     def _sice(self, mfX, fLR=5.0, fSparsity=0.07, nSteps=10000):
         mfC = self._cov_pool(mfX)
         mfC=mfC/torch.diagonal(mfC, dim1=-2, dim2=-1).sum(-1).view(-1,1,1)
         I = 1e-10+1e-9*torch.diag(torch.rand(mfC.shape[1],device = mfC.device)).view(1, mfC.shape[1], mfC.shape[2]).repeat(mfC.shape[0],1,1).type(mfC.dtype)
         zz=self._inv_sqrtm(mfC+I, 7)

         mfInvC=zz.bmm(zz)

         mfCov=mfC*1.0
         mfLLT=mfInvC*1.0 #+1

         mfCov=mfCov
         mfLLT=mfLLT
         mfLLT_prev=1e10*torch.ones(mfLLT.size(), device=mfC.device)
         
         nCounter=0
         for i in range(nSteps):
             mfLLT_plus = torch.relu(mfLLT)
             mfLLT_minus = torch.relu(-mfLLT)
         
             zz = self._inv_sqrtm(mfLLT+I, 7)
             mfGradPart1=-zz.bmm(zz)
            
             mfGradPart2 = 0.5*(mfCov.transpose(1,2) + mfCov)
             mfGradPart12 = mfGradPart1+mfGradPart2
            
             mfGradPart3_plus = mfGradPart12 + fSparsity
             mfGradPart3_minus = -mfGradPart12 + fSparsity
            
             fDec=(1-i/(nSteps-1.0) )
            
             mfLLT_plus = mfLLT_plus - fLR*fDec*mfGradPart3_plus
             mfLLT_minus = mfLLT_minus - fLR*fDec*mfGradPart3_minus
            
             mfLLT_plus = torch.relu(mfLLT_plus)
             mfLLT_minus = torch.relu(mfLLT_minus)
            
             mfLLT = mfLLT_plus-mfLLT_minus 
             mfLLT = 0.5*(mfLLT+mfLLT.transpose(1,2))
            
             fSolDiff = (mfLLT-mfLLT_prev).abs().mean()
             fSparseCount = ((mfLLT.abs()>2e-8)*1.0).mean()

             mfLLT_prev = mfLLT*1.0
             mfLLT_prev = mfLLT_prev
         mfOut = mfLLT
         mfOut = mfOut/torch.sqrt(torch.diagonal(mfOut, dim1=-2, dim2=-1).sum(-1)).view(-1,1,1) #works better and faster convergence
         return mfOut
    
     def _triuvec(self, x):
         return Triuvec.apply(x)     
     

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._sice(x, fLR=self.learingRate, fSparsity=self.sparsity, nSteps=self.iterNum)
         if self.is_vec:
             x = self._triuvec(x)
         return x



class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device).type(dtype)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1).type(dtype)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZYZ = 0.5 * (I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:])).bmm(Z[:,iterN-2,:,:])
         y = ZYZ * torch.pow(normA,-0.5).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZYZ, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.pow(normA, -0.5).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = -0.5*torch.pow(normA, -1.5)*((grad_output*ZY).sum(dim=1).sum(dim=1))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            dldZ = 0.5*((I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])).bmm(der_postCom) -
                          der_postCom.bmm(Z[:,iterN-2,:,:]).bmm(Y[:,iterN-2,:,:]))
            dldY = -0.5*Z[:,iterN-2,:,:].bmm(der_postCom).bmm(Z[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         der_NSiter = der_NSiter.transpose(1, 2)
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag().type(dtype)
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def InvcovpoolLayer(var):
    return InverseCOV.apply(var)

def TriuvecLayer(var):
    return Triuvec.apply(var)
