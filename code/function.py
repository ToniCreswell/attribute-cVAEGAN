import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import make_grid, save_image

import os
from os.path import join

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np


####### Losses #######
def vae_loss_fn(rec_x, x, mu, logVar):
	#Total loss is BCE(x, rec_x) + KL
	BCE = F.binary_cross_entropy(rec_x, x, size_average=False)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
	#(might be able to use nn.NLLLoss2d())
	KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar) #0.5 * sum(1 + log(var) - mu^2 - var)
	return BCE / (x.size(2) ** 2),  KL / mu.size(1)

def class_loss_fn(pred, target):
	loss = F.nll_loss(pred, target)
	return loss

####### Saving outputs/inputs #######
def make_new_folder(exDir):
	i=1
	while os.path.isdir(join(exDir,'Ex_'+str(i))):
		i+=1

	os.mkdir(join(exDir,'Ex_'+str(i)))
	return join(exDir,'Ex_'+str(i))

def plot_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	fig1 = plt.figure()
	assert epochs > 0
	for key in losses:
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor,losses[key], label=key)

	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.title(title)
	fig1.savefig(join(exDir, title+'_plt.png'))

def plot_norm_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	assert epochs > 0
	fig1 = plt.figure()
	for key in losses:
		y = losses[key]
		y -= np.mean(y)
		y /= ( np.std(y) + 1e-6 ) 
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor,y, label=key)
	plt.xlabel('epoch')
	plt.ylabel('normalised loss')
	plt.legend()
	fig1.savefig(join(exDir, 'norm_'+title+'_plt.png'))

def is_ready_to_stop_pretraining(e, epochs):
	return (e >= epochs)

def save_input_args(exDir, opts):
	#save the input args to 
	f = open(join(exDir,'opts.txt'),'w')
	saveOpts =''.join(''.join(str(opts).split('(')[1:])\
		.split(')')[:-1])\
		.replace(',','\n')
	f.write(saveOpts)
	f.close()

def sample_z(batch_size, nz, useCUDA):
	if useCUDA:
		return Variable(torch.randn(batch_size, nz).cuda())
	else:
		return Variable(torch.randn(batch_size, nz))

def label_switch(x,y,cvae,exDir=None):
	print 'switching label...'
	#get x's that have no smile

	if (y.data == 0).all(): #if no samples with label 1 use all samples
		x0 = x
	else:
		zeroIdx = torch.nonzero(y.data)
		x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

	#get z
	mu, logVar, y = cvae.encode(x0)
	z = cvae.re_param(mu, logVar)

	ySmile = Variable(torch.eye(2)[torch.LongTensor(np.ones(y.size(0), dtype=int))]).type_as(z)
	smileSamples = cvae.decode(ySmile, z).cpu()
	

	yNoSmile = Variable(torch.eye(2)[torch.LongTensor(np.zeros(y.size(0), dtype=int))]).type_as(z)
	noSmileSamples = cvae.decode(yNoSmile, z).cpu()
	
	if exDir is not None:
		print 'saving rec w/ and w/out label switch to', join(exDir,'rec_0.png'),'... '
		save_image(x0.data, join(exDir, 'original.png'))
		save_image(smileSamples.data, join(exDir,'rec_1.png'))
		save_image(noSmileSamples.data, join(exDir,'rec_0.png'))


def label_switch_1(x,y,cvae,exDir=None): #when y is a unit not a vector
	print 'switching label...'
	#get x's that have smile
	print type(x)
	if (y.data == 0).all(): #if no samples with label 1 use all samples
		x0 = Variable(x)
	else:
		zeroIdx = torch.nonzero(y.data)
		x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

	#get z
	mu, logVar, y = cvae.encode(x0)
	z = cvae.re_param(mu, logVar)

	ySmile = Variable(torch.LongTensor(np.ones(y.size(), dtype=int))).type_as(z)
	smileSamples = cvae.decode(ySmile, z)
	

	yNoSmile = Variable(torch.LongTensor(np.zeros(y.size(), dtype=int))).type_as(z)
	noSmileSamples = cvae.decode(yNoSmile, z)
	
	if exDir is not None:
		print 'saving rec w/ and w/out label switch to', join(exDir,'rec.png'),'... '
		save_image(x0.data, join(exDir, 'original.png'))
		save_image(smileSamples.cpu().data, join(exDir,'rec_1.png'))
		save_image(noSmileSamples.cpu().data, join(exDir,'rec_0.png'))

	return smileSamples, noSmileSamples

def soft_label_switch_1(x,y,cvae,exDir=None, l0=0, l1=1): #when y is a unit not a vector
	print 'switching label...'
	#get x's that have smile
	zeroIdx = torch.nonzero(y.data)
	x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

	#get z
	mu, logVar, y = cvae.encode(x0)
	z = cvae.re_param(mu, logVar)

	ySmile = Variable(torch.Tensor(y.size()).fill_(l1)).type_as(z)
	smileSamples = cvae.decode(ySmile, z)
	
	yNoSmile = Variable(torch.Tensor(y.size()).fill_(l0)).type_as(z)
	noSmileSamples = cvae.decode(yNoSmile, z)
	
	if exDir is not None:
		print 'saving rec w/ and w/out label switch to', join(exDir,'rec.png'),'... '
		save_image(x0.data, join(exDir, 'original.png'))
		save_image(smileSamples.cpu().data, join(exDir,'rec_'+str(l1)+'.png'))
		save_image(noSmileSamples.cpu().data, join(exDir,'rec_'+str(l0)+'.png'))


def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size(0),1).type_as(x)
	return x,y

def rand_one_hot(noSamples, noLabels, useCUDA):

	y = Variable(torch.eye(noLabels)[torch.LongTensor(np.random.randint(0,noLabels,noSamples, dtype=int))]).type(torch.FloatTensor)
	if useCUDA:
		return y.cuda()
	else:
		return y

def binary_class_score(pred, target, thresh=0.5):
	predLabel = torch.gt(pred, thresh)
	classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
	return  classScoreTest.float().sum()/target.size(0)


