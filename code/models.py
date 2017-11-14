import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np

# from functions import onehot

import os
from os.path import join

class VAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, sig=1):  #sig is the std of the prior
		super(VAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sig = sig

		inSize = imSize / ( 2 ** 4)
		self.inSize = inSize

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)

		self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)

		self.dec1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	
		self.useCUDA = torch.cuda.is_available()

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		mu = self.encMu(x)  #no relu - mean may be zero
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		
		return mu, log_var

	def sample_z(self, noSamples, sig=1):
		z =  sig * torch.randn(noSamples, self.nz)
		if self.useCUDA:
			return Variable(z.cuda())
		else:
			return Variable(z)

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		if self.useCUDA:
			eps = Variable(torch.randn(sigma.size(0), self.nz).cuda())
		else: eps = Variable(torch.randn(sigma.size(0), self.nz))
		
		return mu + self.sig * sigma * eps  #eps.mul(simga)._add(mu)


	def decode(self, z):
		#define the decoder here
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2(z))
		z = F.relu(self.dec3(z))
		z = F.relu(self.dec4(z))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x):
		# the outputs needed for training
		mu, log_var = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(z)

		return reconstruction, mu, log_var

	def loss(self, rec_x, x, mu, logVar):
		#Total loss is BCE(x, rec_x) + KL
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		#(might be able to use nn.NLLLoss2d())
		KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		return BCE / (x.size(2) ** 2),  KL / mu.size(1)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'vae_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'vae_params')))


class GMMAE(nn.Module):
	''' Sparse encoding p, as mixture co-efficients of Gaussians '''

	def __init__(self, nz, np, imSize, fSize=2):  #sig is the std of the prior
		super(GMMAE, self).__init__()
		#define layers here
		self.useCUDA = torch.cuda.is_available()

		self.fSize = fSize
		self.nz = nz
		self.np = np
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)
		self.inSize = inSize

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.p = nn.Linear((fSize * 8) * inSize * inSize, np)

		if  self.useCUDA:
			self.MU = torch.nn.Parameter(torch.randn(np,nz).cuda(), requires_grad=True)
			self.LOGVAR = torch.nn.Parameter(torch.randn(np,nz).cuda(), requires_grad=True)
		else:
			self.MU = torch.nn.Parameter(torch.randn(np,nz), requires_grad=True)
			self.LOGVAR = torch.nn.Parameter(torch.randn(np,nz), requires_grad=True)

		self.register_parameter(name='mu', param=self.MU)
		self.register_parameter(name='logvar', param=self.LOGVAR)

		self.dec1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		p = F.softmax(self.p(x))

		mu = torch.matmul(p, self.MU)
		log_var = torch.matmul(p, self.LOGVAR)
		
		return p, self.re_param(mu, log_var)

	def sample_z(self, noSamples):
		#p ~ cat(1/np)
		if self.useCUDA:
			p = Variable(torch.LongTensor(np.random.randint(0, self.np, size=(noSamples)))).cuda()
		else:
			p = torch.LongTensor(np.random.randint(0, self.np, size=(noSamples)))
		mu = self.MU[p]
		log_var = self.LOGVAR[p]
		
		return self.re_param(mu, log_var)


	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		if self.useCUDA:
			eps = Variable(torch.randn(sigma.size(0), self.nz).cuda())
		else: eps = Variable(torch.randn(sigma.size(0), self.nz))
		
		return mu + sigma * eps  #eps.mul(simga)._add(mu)


	def decode(self, z):
		#define the decoder here
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2(z))
		z = F.relu(self.dec3(z))
		z = F.relu(self.dec4(z))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x):
		# the outputs needed for training
		p, z = self.encode(x)
		reconstruction = self.decode(z)

		return p, reconstruction

	def loss(self, rec_x, x, p):
		#Total loss is BCE(x, rec_x) + penalty on p to be sparse + (fisher)
		regP = torch.norm(p, p=2)
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		return BCE / (x.size(2) ** 2), regP

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'gmmae_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'gmmae_params')))

class DISCRIMINATOR(nn.Module):

	def __init__(self, imSize, fSize=2, numLabels=1):
		super(DISCRIMINATOR, self).__init__()
		#define layers here

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)
		self.numLabels = numLabels

		self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, numLabels)
	
		self.useCUDA = torch.cuda.is_available()

	def discriminate(self, x):
		x = F.relu(self.dis1(x))
		x = F.relu(self.dis2(x))
		x = F.relu(self.dis3(x))
		x = F.relu(self.dis4(x))
		x = x.view(x.size(0), -1)
		if self.numLabels == 1:
			x = F.sigmoid(self.dis5(x))
		else:
			x = F.softmax(self.dis5(x))
		
		return x

	def forward(self, x):
		# the outputs needed for training
		return self.discriminate(x)


	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir,'class_dis_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir,'class_dis_params')))

class DISCRIMINATORGRAY(nn.Module):

	def __init__(self, imSize, fSize=2):
		super(DISCRIMINATORGRAY, self).__init__()
		#define layers here

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)

		self.dis1 = nn.Conv2d(1, fSize, 5, stride=2, padding=2)
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, 1)
	
		self.useCUDA = torch.cuda.is_available()

	def discriminate(self, x):
		x = F.relu(self.dis1(x))
		x = F.relu(self.dis2(x))
		x = F.relu(self.dis3(x))
		x = F.relu(self.dis4(x))
		x = x.view(x.size(0), -1)
		x = F.sigmoid(self.dis5(x))
		
		return x


	def forward(self, x):
		# the outputs needed for training
		return self.discriminate(x)


	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir,'class_dis_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir,'class_dis_params')))

class CVAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, numLabels=2):
		super(CVAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize

		inSize = imSize / (2 ** 4)
		self.inSize = inSize
		self.numLabels = numLabels

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)

		self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encY = nn.Linear((fSize * 8) * inSize * inSize, numLabels)

		self.dec1 = nn.Linear(nz+numLabels, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.dec2b = nn.BatchNorm2d(fSize * 4)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec3b = nn.BatchNorm2d(fSize * 2)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec4b = nn.BatchNorm2d(fSize)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	
		self.useCUDA = torch.cuda.is_available()

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		mu = self.encMu(x)  #no relu - mean may be negative
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		y = F.softmax(self.encY(x.detach()))
		
		return mu, log_var, y

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		if self.useCUDA:
			eps = Variable(torch.randn(sigma.size(0), self.nz).cuda())
		else: eps = Variable(torch.randn(sigma.size(0), self.nz))
		
		return mu + sigma * eps  #eps.mul(simga)._add(mu)

	def decode(self, y, z):
		#define the decoder here
		z = torch.cat([y,z], dim=1)
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2b(self.dec2(z)))
		z = F.relu(self.dec3b(self.dec3(z)))
		z = F.relu(self.dec4b(self.dec4(z)))
		z = F.sigmoid(self.dec5(z))

		return z

	def loss(self, rec_x, x, mu, logVar):
		#Total loss is BCE(x, rec_x) + KL
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		#(might be able to use nn.NLLLoss2d())
		KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		return BCE / (x.size(2) ** 2),  KL / mu.size(1)

	def forward(self, x):
		# the outputs needed for training
		mu, log_var, y = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(y, z)

		return reconstruction, mu, log_var, y

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'cvae_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'cvae_params')))

class CVAE1(nn.Module):

	def __init__(self, nz, imSize, fSize=2, sig=1):
		super(CVAE1, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sig = sig

		inSize = imSize / (2 ** 4)
		self.inSize = inSize

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)

		self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encY = nn.Linear((fSize * 8) * inSize * inSize, 1)

		self.dec1 = nn.Linear(nz+1, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1 ,output_padding=1)
		self.dec2b = nn.BatchNorm2d(fSize * 4)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec3b = nn.BatchNorm2d(fSize * 2)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec4b = nn.BatchNorm2d(fSize)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	
		self.useCUDA = torch.cuda.is_available()

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		mu = self.encMu(x)  #no relu - mean may be negative
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		y = F.sigmoid(self.encY(x.detach()))
		
		return mu, log_var, y

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		if self.useCUDA:
			eps = Variable(torch.randn(sigma.size(0), self.nz).cuda())
		else: eps = Variable(torch.randn(sigma.size(0), self.nz))
		
		return mu + sigma * eps  #eps.mul(simga)._add(mu)

	def sample_z(self, noSamples, sig=1):
		z =  sig * torch.randn(noSamples, self.nz)
		if self.useCUDA:
			return Variable(z.cuda())
		else:
			return Variable(z)

	def decode(self, y, z):
		#define the decoder here
		z = torch.cat([y,z], dim=1)
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2b(self.dec2(z)))
		z = F.relu(self.dec3b(self.dec3(z)))
		z = F.relu(self.dec4b(self.dec4(z)))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x):
		# the outputs needed for training
		mu, log_var, y = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(y, z)

		return reconstruction, mu, log_var ,y

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'cvae1_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'cvae1_params')))

	def loss(self, rec_x, x, mu, logVar):
		sigma2 = Variable(torch.Tensor([self.sig]))
		if self.useCUDA:
			sigma2 = sigma2.cuda()
		logVar2 = torch.log(sigma2)
		#Total loss is BCE(x, rec_x) + KL
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		#(might be able to use nn.NLLLoss2d())
		if self.sig == 1:
			KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		else:
			KL = 0.5 * torch.sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
		return BCE / (x.size(2) ** 2),  KL / mu.size(1)

class CVAEGRAY(nn.Module):

	def __init__(self, nz, imSize, numLabels=7, fSize=2, sig=1):
		super(CVAEGRAY, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sig = sig

		inSize = imSize / (2 ** 4)
		self.inSize = inSize

		self.enc1 = nn.Conv2d(1, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)

		self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)
		self.encY = nn.Linear((fSize * 8) * inSize * inSize, numLabels)

		self.dec1 = nn.Linear(nz+numLabels, (fSize) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize, fSize * 4, 3, stride=2, padding=1 ,output_padding=1)
		self.dec2b = nn.BatchNorm2d(fSize * 4)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec3b = nn.BatchNorm2d(fSize * 2)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec4b = nn.BatchNorm2d(fSize)
		self.dec5 = nn.ConvTranspose2d(fSize, 1, 3, stride=2, padding=1, output_padding=1)

	
		self.useCUDA = torch.cuda.is_available()

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		mu = self.encMu(x)  #no relu - mean may be negative
		log_var = self.encLogVar(x) #no relu - log_var may be negative
		y = F.softmax(self.encY(x.detach()))
		
		return mu, log_var, y

	def re_param(self, mu, log_var):
		#do the re-parameterising here
		sigma = torch.exp(log_var/2)  #sigma = exp(log_var/2) #torch.exp(log_var/2)
		if self.useCUDA:
			eps = Variable(torch.randn(sigma.size(0), self.nz).cuda())
		else: eps = Variable(torch.randn(sigma.size(0), self.nz))
		
		return mu + sigma * eps  #eps.mul(simga)._add(mu)

	def sample_z(self, noSamples, sig=1):
		z =  sig * torch.randn(noSamples, self.nz)
		if self.useCUDA:
			return Variable(z.cuda())
		else:
			return Variable(z)

	def decode(self, y, z):
		#define the decoder here
		z = torch.cat([y,z], dim=1)
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2b(self.dec2(z)))
		z = F.relu(self.dec3b(self.dec3(z)))
		z = F.relu(self.dec4b(self.dec4(z)))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x):
		# the outputs needed for training
		mu, log_var, y = self.encode(x)
		z = self.re_param(mu, log_var)
		reconstruction = self.decode(y, z)

		return reconstruction, mu, log_var ,y

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'cvae1_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'cvae1_params')))

	def loss(self, rec_x, x, mu, logVar):
		sigma2 = Variable(torch.Tensor([self.sig]))
		if self.useCUDA:
			sigma2 = sigma2.cuda()
		logVar2 = torch.log(sigma2)
		#Total loss is BCE(x, rec_x) + KL
		BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		#(might be able to use nn.NLLLoss2d())
		if self.sig == 1:
			KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
		else:
			KL = 0.5 * torch.sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
		return BCE / (x.size(2) ** 2),  KL / mu.size(1)


class AUX(nn.Module):
	#map z to a label, y
	def __init__(self, nz, numLabels=1):
		super(AUX, self).__init__()

		self.nz = nz
		self.numLabels = numLabels

		self.aux1 = nn.Linear(nz, 1000)
		self.aux2 = nn.Linear(1000, 1000)
		self.aux3 = nn.Linear(1000, numLabels)


	def infer_y_from_z(self, z):
		z = F.relu(self.aux1(z))
		z = F.relu(self.aux2(z))
		if self.numLabels==1:
			z = F.sigmoid(self.aux3(z))
		else:
			z = F.softmax(self.aux3(z))

		return z

	def forward(self, z):
		return self.infer_y_from_z(z)

	def loss(self, pred, target):
		return F.nll_loss(pred, target)


	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'aux_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'aux_params')))

class DIS_X_X(nn.Module):

	def __init__(self, imSize, fSize=2):
		super(DIS_X_X, self).__init__()
		#define layers here

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)

		self.dis1 = nn.Conv2d(6, fSize, 5, stride=2, padding=2)  #6 cause 2 images are concatenated in the feature channel
		self.dis1b = nn.BatchNorm2d(fSize)
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis2b = nn.BatchNorm2d(fSize * 2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis3b = nn.BatchNorm2d(fSize * 4)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis4b = nn.BatchNorm2d(fSize * 8)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, 1)
	
		self.useCUDA = torch.cuda.is_available()

	def discriminate(self, x1, x2):
		# x = torch.cat([x1,x2], dim=1)
		# x = F.relu(self.dis1b(self.dis1(x)))
		# x = F.relu(self.dis2b(self.dis2(x)))
		# x = F.relu(self.dis3b(self.dis3(x)))
		# x = F.relu(self.dis4b(self.dis4(x)))
		# x = x.view(x.size(0), -1)
		# x = self.dis5(x)

		x = torch.cat([x1,x2], dim=1)
		x = F.relu(self.dis1(x))
		x = F.relu(self.dis2(x))
		x = F.relu(self.dis3(x))
		x = F.relu(self.dis4(x))
		x = x.view(x.size(0), -1)
		x = self.dis5(x)
		
		return x


	def forward(self, x1, x2):
		# the outputs needed for training
		return self.discriminate(x1, x2)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dis_x_x_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dis_x_x_params')))

#################################### WGAN #########################################

class WGEN(nn.Module):

	def __init__(self, imSize, nz=100, prior=torch.randn, fSize=2):
		super(WGEN, self).__init__()

		self.nz = nz
		self.prior = prior

		inSize = imSize // (2 ** 4)
		self.inSize = inSize

		self.useCUDA = torch.cuda.is_available()

		self.gen1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.gen2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.gen2b = nn.BatchNorm2d(fSize * 4)
		self.gen3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.gen3b = nn.BatchNorm2d(fSize * 2)
		self.gen4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.gen4b = nn.BatchNorm2d(fSize)
		self.gen5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	def sample_z(self, no_samples, prior=None):
		if prior is None:
			prior = self.prior
		return prior(no_samples, self.nz)

	def gen(self, z):
		x = F.relu(self.gen1(z))
		x = x.view(z.size(0), -1, self.inSize, self.inSize)
		x = F.relu(self.gen2b(self.gen2(x)))
		x = F.relu(self.gen3b(self.gen3(x)))
		x = F.relu(self.gen4b(self.gen4(x)))
		x = F.sigmoid(self.gen5(x))

		return x

	def forward(self, z):
		return self.gen(z)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'wgen_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'wgen_params')))
		self = torch.load(join(exDir,'gen_params'))

class WDIS(nn.Module):
	def __init__(self, imSize, fSize=2):
		super(WDIS, self).__init__()

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)



		self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)  #6 cause 2 images are concatenated in the feature channel
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, 1)
	
		self.useCUDA = torch.cuda.is_available()


	def dis(self, x):

		lrelu = torch.nn.LeakyReLU(0.2)

		d = lrelu(self.dis1(x))
		d = lrelu(self.dis2(d))
		d = lrelu(self.dis3(d))
		d = lrelu(self.dis4(d))
		d = d.view(x.size(0), -1)
		d = self.dis5(d)  #no sigmoid cause is a WGAN

		return d

	def forward(self, x):
		return self.dis(x)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'wdis_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'wdis_params')))

#################################### DCGAN #########################################

class dcGEN(nn.Module):

	def __init__(self, imSize, nz=100, prior=torch.randn, fSize=2):
		super(dcGEN, self).__init__()

		self.nz = nz
		self.prior = prior

		inSize = imSize // (2 ** 4)
		self.inSize = inSize

		self.useCUDA = torch.cuda.is_available()

		self.gen1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.gen2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.gen2b = nn.BatchNorm2d(fSize * 4)
		self.gen3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.gen3b = nn.BatchNorm2d(fSize * 2)
		self.gen4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.gen4b = nn.BatchNorm2d(fSize)
		self.gen5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	def sample_z(self, no_samples, prior=None):
		if prior is None:
			prior = self.prior
		return prior(no_samples, self.nz)

	def gen(self, z):

		lrelu = torch.nn.LeakyReLU(0.2)

		x = lrelu(self.gen1(z))
		x = x.view(z.size(0), -1, self.inSize, self.inSize)
		x = lrelu(self.gen2b(self.gen2(x)))
		x = lrelu(self.gen3b(self.gen3(x)))
		x = lrelu(self.gen4b(self.gen4(x)))
		x = F.tanh(self.gen5(x))

		return x

	def forward(self, z):
		return self.gen(z)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'gen_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'gen_params')))

class dcDIS(nn.Module):
	def __init__(self, imSize, fSize=2):
		super(dcDIS, self).__init__()

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)

		self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)  #6 cause 2 images are concatenated in the feature channel
		self.dis1b = nn.BatchNorm2d(fSize)
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis2b = nn.BatchNorm2d(fSize * 2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis3b = nn.BatchNorm2d(fSize * 4)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis4b = nn.BatchNorm2d(fSize * 8)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, 1)
	
		self.useCUDA = torch.cuda.is_available()


	def dis(self, x):

		lrelu = torch.nn.LeakyReLU(0.2)

		d = lrelu(self.dis1b(self.dis1(x)))
		d = lrelu(self.dis2b(self.dis2(d)))
		d = lrelu(self.dis3b(self.dis3(d)))
		d = lrelu(self.dis4b(self.dis4(d)))
		d = d.view(x.size(0), -1)
		d = F.sigmoid(self.dis5(d)) #no sigmoid cause is a WGAN

		return d

	def forward(self, x):
		return self.dis(x)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dis_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dis_params')))

	def ones(self, N):
		ones = Variable(torch.Tensor(N,1).fill_(1))
		if self.useCUDA:
			return ones.cuda()
		else:
			return ones

	def zeros(self, N):
		zeros = Variable(torch.Tensor(N,1).fill_(0))
		if self.useCUDA:
			return zeros.cuda()
		else:
			return zeros

 
class AlexNet(nn.Module):

	def __init__(self, imSize):
		super(AlexNet, self).__init__()

		self.l1 = nn.Conv2d(3, 48, 3, stride=1)
		self.l2 = nn.MaxPool2d(2)
		self.l3 = nn.Conv2d(48, 128, 3, stride=1)
		self.l4 = nn.MaxPool2d(2)
		self.l5 = nn.Conv2d(128, 192, 3, stride=1)
		self.l6 = nn.Conv2d(192, 192, 3, stride=1)
		self.l7 = nn.Conv2d(192, 128, 3, stride=1)
		self.l8 = nn.MaxPool2d(2)
		#reshape
		self.l9 = nn.Linear(128 * (imSize//4), 2048)
		self.l10 = nn.Linear(2048, 2048)
		self.l11 = nn.Linear(2048, 30)

	def forward(self, x):
		print 'need to check no of labels'
		assert False
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = F.relu(self.l4(x))
		x = F.relu(self.l5(x))
		x = F.relu(self.l6(x))
		x = F.relu(self.l7(x))
		x = F.relu(self.l8(x))
		x = x.view(x.size(0), -1)
		print x.size()
		x = F.relu(self.l9(x))
		x = F.relu(self.l10(x))
		x = F.relu(self.l11(x))
		x = F.sigmoid(x)

		return x

	def loss(self, predY, Y):
		lossFn = nn.BCELoss()
		return lossFn(predY, Y.type_as(predY))

	def predLabels(self, predY, thresh=0.5): #not accurate!
		return torch.ge(predY, thresh)

	def acc(self, predY, Y, thresh=0.5):
		predLabels = self.predLabels(predY, thresh)
		noCorrect = torch.eq(predLabels, Y)
		acc = torch.mean(noCorrect)
		return acc


	def changes(labels1, labels2):
		SMILEIDX = 31
		noIdx = len(labels1)
		labels1 = labels1[:31,]
		#remove the smile label (label no. )















