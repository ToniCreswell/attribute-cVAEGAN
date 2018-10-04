import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np

# from functions import onehot

import os
from os.path import join


class DISCRIMINATOR(nn.Module):

	def __init__(self, imSize, fSize=2, numLabels=1):
		super(DISCRIMINATOR, self).__init__()
		#define layers here

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize // ( 2 ** 4)
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



class CVAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, numLabels=2):
		super(CVAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize

		inSize = imSize // (2 ** 4)
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

		inSize = imSize // (2 ** 4)
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
















