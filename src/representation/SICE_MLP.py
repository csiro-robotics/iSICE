'''
@file: MPNCOV.py
@author: Jiangtao Xie
@author: Peihua Li
Please cite the paper below if you use the code:

Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.

Peihua Li, Jiangtao Xie, Qilong Wang and Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition? IEEE Int. Conf. on Computer Vision (ICCV),  pp. 2070-2078, 2017.

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
'''
import torch
import torch.nn as nn
from torch.autograd import Function
from numpy import linalg as LA
import numpy as np

class SICE_MLP(nn.Module):
     """Matrix power normalized Covariance pooling (MPNCOV)
        implementation of fast MPN-COV (i.e.,iSQRT-COV)
        https://arxiv.org/abs/1712.01034

     Args:
         iterNum: #iteration of Newton-schulz method
         is_sqrt: whether perform matrix square root or not
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=None, sparsity_val=0.0, sice_lrate=0.0):

         super(SICE_MLP, self).__init__()
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
         self.mlp_sp_1 = nn.Sequential(
             nn.Linear(784, 392),
             nn.ReLU(inplace=True)
             )
         self.mlp_sp_2 = nn.Sequential(
             nn.Linear(392, 1)
             )
         self.mlp_lr_fc1 = nn.Sequential(
             nn.Linear(784, 392),
             nn.ReLU(inplace=True)
             )
         self.mlp_lr_fc2 = nn.Sequential(
             nn.Linear(392, 1)
             )
        #  self.mlp_sp_1 = nn.Sequential(
        #      nn.Linear(196, 98),
        #      nn.ReLU(inplace=True)
        #      )
        #  self.mlp_sp_2 = nn.Sequential(
        #      nn.Linear(98, 1)
        #      )
        #  self.mlp_lr_fc1 = nn.Sequential(
        #      nn.Linear(196, 98),
        #      nn.ReLU(inplace=True)
        #      )
        #  self.mlp_lr_fc2 = nn.Sequential(
        #      nn.Linear(98, 1)
        #      )
        #  self.mlp_sp_1 = nn.Sequential(
        #      nn.Linear(49, 25),
        #      nn.ReLU(inplace=True)
        #      )
        #  self.mlp_sp_2 = nn.Sequential(
        #      nn.Linear(25, 1)
        #      )
        #  self.mlp_lr_fc1 = nn.Sequential(
        #      nn.Linear(49, 25),
        #      nn.ReLU(inplace=True)
        #      )
        #  self.mlp_lr_fc2 = nn.Sequential(
        #      nn.Linear(25, 1)
        #      )
         self.sice_lr = 0
         self.sice_sparsity = 0

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
         
    #  def _inv_sqrtm(self, x, iterN):
    #      batchSize = x.shape[0]
    #      dim = x.shape[1]
    #      dtype = x.dtype
    #      I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    #      normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
    #      A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
    #      ZY = 0.5 * (I3 - A)
    #      if iterN < 2:
    #          ZY = 0.5*(I3 - A)
    #          YZY = A.bmm(ZY)
    #      else:
    #          Y = A.bmm(ZY)
    #          Z = ZY
    #          for _ in range(iterN - 2):
    #              ZY = 0.5 * (I3 - Z.bmm(Y))
    #              Y = Y.bmm(ZY)
    #              Z = ZY.bmm(Z)
    #          #YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
    #          YZY = 0.5 * (I3 - Z.bmm(Y)).bmm(Z) #to compute inv
    #          del Y,Z,ZY,I3
    #      #z = Z/torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
    #      z = YZY * torch.pow(normA,-0.5).view(batchSize, 1, 1).expand_as(x) #to compute inv
    #      return z
     def _sice_full(self, mfX, fLR=5.0, fSparsity=0.07, nSteps=10000):
         #batchSize, dim, h, w = mfX.data.shape
         #x = mfX.reshape(batchSize, dim, h * w)
         #mfC = 1. / (h * w) * x.bmm(x.transpose(1, 2))
         mfC = self._cov_pool(mfX)
         #norm_ = torch.diagonal(mfC, dim1=-2, dim2=-1).sum(-1)
         mfC=mfC/torch.diagonal(mfC, dim1=-2, dim2=-1).sum(-1).view(-1,1,1)
         I = 1e-10+1e-9*torch.diag(torch.rand(mfC.shape[1],device = mfC.device)).view(1, mfC.shape[1], mfC.shape[2]).repeat(mfC.shape[0],1,1).type(mfC.dtype)
         zz=self._inv_sqrtm(mfC+I, 7)

         mfInvC=zz.bmm(zz)

         mfCov=mfC*1.0
         mfLLT=mfInvC*1.0 #+1

         mfCov=mfCov#.detach()
         mfLLT=mfLLT#.detach()  
         mfLLT_prev=1e10*torch.ones(mfLLT.size(), device=mfC.device)#.detach()
         
         nCounter=0
         for i in range(nSteps):
             mfLLT_plus=torch.relu(mfLLT)
             mfLLT_minus=torch.relu(-mfLLT)
             while True:
                 try:
                     zz=self._inv_sqrtm(mfLLT+I, 7) #added by me
                     mfGradPart1=-zz.bmm(zz)
                 except:
                     I3 = 1e-10+1e-9* torch.diag(torch.rand(mfLLT.shape[1], device=mfC.device)).view(1, mfLLT.shape[1], mfLLT.shape[1]).repeat(mfLLT.shape[0], 1, 1).type(mfLLT.dtype)
                     mfLLT=mfLLT+I3
                 else:
                     break
            
             mfGradPart2=0.5*( mfCov.transpose(1,2) + mfCov )
             mfGradPart12=mfGradPart1+mfGradPart2
            
             mfGradPart3_plus=mfGradPart12 + fSparsity #*mfLLT_plus.sign()
             #mfGradPart3_plus=mfGradPart12 + fSparsity - (fSparsity* torch.diag(torch.ones(mfLLT.shape[1], device=mfLLT.device)).repeat(mfLLT.shape[0],1,1))
             mfGradPart3_minus=-mfGradPart12 + fSparsity #*mfLLT_minus.sign()
            
             fDec=(1-i/(nSteps-1.0) )
            
             mfLLT_plus=mfLLT_plus - fLR*fDec*mfGradPart3_plus
             mfLLT_minus=mfLLT_minus - fLR*fDec*mfGradPart3_minus
            
             mfLLT_plus=torch.relu(mfLLT_plus)
             mfLLT_minus=torch.relu(mfLLT_minus)
            
             mfLLT=mfLLT_plus-mfLLT_minus 
             mfLLT=0.5*(mfLLT+mfLLT.transpose(1,2))# + I3 
            
             fSolDiff=(mfLLT-mfLLT_prev).abs().mean()
             fSparseCount=((mfLLT.abs()>2e-8)*1.0).mean()

             mfLLT_prev=mfLLT*1.0
             mfLLT_prev=mfLLT_prev#.detach()

            #  torch.set_printoptions(edgeitems=3, sci_mode=False, linewidth=200)
            #  print(mfLLT[0],'--> iteration:', i, ', Sparse count: ', round(fSparseCount.item(),4), ', SolDiff: ',round(fSolDiff.item(),4))
            #  torch.set_printoptions(profile="default", edgeitems=3, sci_mode=None, linewidth=80)
            
             if fSolDiff<0.001 and fSparseCount<0.7:
                 nCounter=nCounter+1
             if nCounter>3:
                 print('Number of iterations: ', i)
                 print('Sparsity count: ', fSparseCount)
                 break
         #mfLLT = mfLLT/torch.sqrt(torch.diagonal(mfLLT, dim1=-2, dim2=-1).sum(-1)).view(-1,1,1) #trace norm
         mfOut=mfInvC- (mfInvC-mfLLT)#.detach() #fast method
         #exit()
         #mfOut = mfLLT
         mfOut = mfOut/torch.sqrt(torch.diagonal(mfOut, dim1=-2, dim2=-1).sum(-1)).view(-1,1,1) #trace norm fast norm
         return mfOut
     
     def _adaptive_sigmoid(self, x, offset, beta, max_range):
         #return (limit/(1+torch.exp(-torch.exp(torch.tensor(1.))*x+torch.exp(torch.tensor(2.))))).reshape(-1,1,1)
         #return (offset + ((1/torch.exp(-beta*x))-0.5)*2*range_).reshape(-1,1,1)
         #x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
         return (max_range/(1+torch.exp(-beta*(x-offset)))).reshape(-1,1,1)

     def _triuvec(self, x):
         return Triuvec.apply(x)     
     

    #  def forward(self, x):
    #      if self.dr is not None:
    #          x = self.conv_dr_block(x)
    #      x_cp = x.mean(dim=1).reshape(x.shape[0],-1)
    #      #x_ = self.mlp_2(self.mlp_1(x.mean(dim=1).reshape(x.shape[0],-1)) + x_cp)
    #      x_ = self.mlp_3(self.mlp_2(self.mlp_1(x_cp)))
    #      #x_ = self.mlp_1(x.mean(dim=1).reshape(x.shape[0],-1))
    #      #x_ = self.sparsity/(1.+torch.exp(-torch.exp(torch.tensor(1.))*x_+torch.exp(torch.tensor(2.)))) #sigmoid
    #      x__ = self._adaptive_sigmoid(x_,self.sparsity)
    #      #print('Sparsity before activation:',x_.mean().item(), ', sparsity after activation: ', x__.mean().item())
    #      #lr_mlp = self.mlp_lr_fc1(x.mean(dim=1).reshape(x.shape[0],-1))
    #      lr_mlp = self.mlp_lr_fc3(self.mlp_lr_fc2(self.mlp_lr_fc1(x_cp)))
    #      #lr_mlp = self.learingRate/(1.+torch.exp(-torch.exp(torch.tensor(1.))*lr_mlp+torch.exp(torch.tensor(2.)))) #sigmoid
    #      lr_mlp_ = self._adaptive_sigmoid(lr_mlp, self.learingRate)
    #      #print('Learning rate before activation:',lr_mlp.mean().item(), ', after activation: ', lr_mlp_.mean().item())
    #      #x = self._sice_full_simplified(x, fLR=lr_mlp.reshape(-1,1,1), fSparsity=x_.reshape(-1,1,1), nSteps=self.iterNum)
    #      #print('[Mean:] learning rate:',lr_mlp.mean().item(), ', sparsity: ', x_.mean().item(), '[Max:] learning rate:',lr_mlp.max().item(), ', sparsity: ', x_.max().item())
    #      x = self._sice_full(x, fLR=lr_mlp_, fSparsity=x__, nSteps=self.iterNum)
    #      self.sice_lr = lr_mlp_.mean()
    #      self.sice_sparsity = x__.mean()
    #      #x = torch.mul(x.sign(), torch.sqrt(x.abs()+1e-8))
    #      if self.is_vec:
    #          x = self._triuvec(x)
    #      #x = nn.functional.normalize(x)
    #      return x

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         
         # mlp input
         x_input = x.mean(dim=1).reshape(x.shape[0],-1)

         #mlp for learning rate
        #  lr_fc1 = self.mlp_lr_fc1(x_input)
        #  lr_mlp = self.mlp_lr_fc4(self.mlp_lr_fc3((self.m(lp_lr_fc2(lr_fc1)+lr_fc1)))
         lr_mlp = self.mlp_lr_fc2(self.mlp_lr_fc1(x_input))
         lr_mlp_ = self._adaptive_sigmoid(lr_mlp, offset=0.0, beta=1.5, max_range=2.0)
         ##print('Learning before activation:',lr_mlp.mean().item(), ', Learning after activation: ', lr_mlp_.mean().item())

         #mlp for sparsity
        #  sp_fc1 = self.mlp_1(x_input)
        #  sp_mlp = self.mlp_4(self.mlp_3((self.mlp_2(sp_fc1)+sp_fc1)))
         sp_mlp = self.mlp_sp_2(self.mlp_sp_1(x_input))
         sp_mlp_ = self._adaptive_sigmoid(sp_mlp, offset=0.005, beta=5.0, max_range=0.09)
         ##print('Sparsity before activation:',sp_mlp.mean().item(), ', Sparsity after activation: ', sp_mlp_.mean().item())

         print('LR [pre-activation: ', lr_mlp.mean().item(), ', post-activation: ', lr_mlp_.mean().item(), '], SP [pre-activation: ', sp_mlp.mean().item(), ', post-activation: ', sp_mlp_.mean().item(), ']')

         #print('[Mean:] learning rate:',lr_mlp.mean().item(), ', sparsity: ', x_.mean().item(), '[Max:] learning rate:',lr_mlp.max().item(), ', sparsity: ', x_.max().item())

         x = self._sice_full(x, fLR=lr_mlp_, fSparsity=sp_mlp_, nSteps=self.iterNum)
         self.sice_lr = lr_mlp_.mean()
         self.sice_sparsity = sp_mlp_.mean()
         #x = torch.mul(x.sign(), torch.sqrt(x.abs()+1e-8))
         if self.is_vec:
             x = self._triuvec(x)
         #x = nn.functional.normalize(x)
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
            #YZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:])) #original
            ZYZ = 0.5 * (I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:])).bmm(Z[:,iterN-2,:,:])
         #y = YZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x) #original
         y = ZYZ * torch.pow(normA,-0.5).view(batchSize, 1, 1).expand_as(x)
         #ctx.save_for_backward(input, A, YZY, normA, Y, Z) #original
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
         #der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x) # original
         der_postCom = grad_output*torch.pow(normA, -0.5).view(batchSize, 1, 1).expand_as(x)
         #der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         der_postComAux = -0.5*torch.pow(normA, -1.5)*((grad_output*ZY).sum(dim=1).sum(dim=1))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            #dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
            #              Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            #dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
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
