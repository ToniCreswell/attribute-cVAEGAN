#conditional VAE+GAN trained on smile/no smile faces -- info seperation!
import sys
sys.path.append('../')

from dataload import CELEBA
from function import make_new_folder, plot_losses, vae_loss_fn, save_input_args, \
is_ready_to_stop_pretraining, sample_z, class_loss_fn, label_switch, plot_norm_losses #, one_hot
from models import CVAE, DISCRIMINATOR

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import os
from os.path import join

import argparse

from PIL import Image

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from time import time

EPSILON = 1e-6

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data/datasets/LabelSwap', type=str)
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--maxEpochs', default=10, type=int)
	parser.add_argument('--nz', default=100, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	parser.add_argument('--outDir', default='../../Experiments/celebA_cVAEGAN', type=str)
	parser.add_argument('--commit', default='None', type=str)
	parser.add_argument('--alpha', default=1e-3, type=float) #weight on the kl term
	parser.add_argument('--gamma', default=1, type=float) #weight on the aux enc loss
	parser.add_argument('--delta', default=1, type=float) #weight on the adversarial loss
	parser.add_argument('--rho', default=1, type=float) #weight on the class loss for the vae
	parser.add_argument('--comments', type=str) #any comments at run time
	parser.add_argument('--mom', default=0.9, type=float) #momentum in rms prop of dis

	return parser.parse_args()


if __name__=='__main__':
	opts = get_args()

	####### Data set #######
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
	trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor())
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor())
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'


	####### Create model #######
	cvae = CVAE(nz=opts.nz, imSize=64, fSize=opts.fSize)
	dis = DISCRIMINATOR(imSize=64, fSize=opts.fSize)

	if cvae.useCUDA:
		print 'using CUDA'
		cvae.cuda()
		dis.cuda()
	else: print '\n *** NOT USING CUDA ***\n'

	print cvae
	print dis



	####### Define optimizer #######
	optimizerCVAE = optim.RMSprop(cvae.parameters(), lr=opts.lr)  #specify the params that are being upated
	optimizerDIS = optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.mom)

	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)  #save training opts


	losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[]}
	Ns = len(trainLoader)*opts.batchSize  #no samples
	Nb = len(trainLoader)  #no batches
	####### Start Training #######
	for e in range(opts.maxEpochs):
		cvae.train()
		dis.train()

		epochLoss = 0
		epochLoss_kl = 0
		epochLoss_bce = 0
		epochLoss_dis = 0
		epochLoss_gen = 0
		epochLoss_class = 0

		TIME = time()

		for i, data in enumerate(trainLoader, 0):
			
			x, y = data
			if cvae.useCUDA:
				x, y = Variable(x.cuda()), Variable(y.cuda())
			else:
				x, y = Variable(x), Variable(y)

			#get ouput, clac loss, calc all grads, optimise
			#VAE loss
			outRec, outMu, outLogVar, predY = cvae(x)
			bceLoss, klLoss = cvae.loss(rec_x=outRec, x=x, mu=outMu, logVar=outLogVar)
			vaeLoss = bceLoss + opts.alpha*klLoss

			#Classification loss  #not on reconstructed sample
			classLoss = class_loss_fn(pred=predY, target=y) 
			vaeLoss += opts.rho * classLoss
			
			#DIS loss
			pXreal = dis(x)
			pXfakeRec = dis(outRec.detach())
			zRand = sample_z(x.size(0), opts.nz, cvae.useCUDA)
			yRand = Variable(torch.eye(2)[torch.LongTensor(y.data.cpu().numpy())]).type_as(zRand)
			pXfakeRand = dis(cvae.decode(yRand, zRand).detach())
			fakeLabel = Variable(torch.Tensor(pXreal.size()).zero_()).type_as(pXreal)
			realLabel = Variable(torch.Tensor(pXreal.size()).fill_(1)).type_as(pXreal)
			disLoss = 0.3 * (bce(pXreal, realLabel, size_average=False) + \
				bce(pXfakeRec, fakeLabel, size_average=False) + \
				bce(pXfakeRand, fakeLabel, size_average=False)) / pXreal.size(1)


			#GEN loss
			pXfakeRec = dis(outRec)
			pXfakeRand = dis(cvae.decode(yRand, zRand))
			genLoss = 0.5 * (bce(pXfakeRec, realLabel,size_average=False) +\
			 bce(pXfakeRand, realLabel, size_average=False)) / pXfakeRec.size(1)

			#include the GENloss (the encoder loss) with the VAE loss
			vaeLoss += opts.delta * genLoss

			#zero the grads - otherwise they will be acculated
			#fill in grads and do updates:
			optimizerCVAE.zero_grad()
			vaeLoss.backward() #fill in grads
			optimizerCVAE.step()

			optimizerDIS.zero_grad()
			disLoss.backward()
			optimizerDIS.step()

			epochLoss += vaeLoss.data[0]
			epochLoss_kl += klLoss.data[0]
			epochLoss_bce += bceLoss.data[0]
			epochLoss_gen += genLoss.data[0]
			epochLoss_dis += disLoss.data[0]
			epochLoss_class += classLoss.data[0]


			if i%100 == 0:
				i+=1
				print '[%d, %d] loss: %0.5f, bce: %0.5f, alpha*kl: %0.5f, gen: %0.5f, dis: %0.5f, class: %0.5f, time: %0.3f' % \
		 (e, i, epochLoss/i, epochLoss_bce/i, opts.alpha*epochLoss_kl/i, epochLoss_gen/i, epochLoss_dis/i, \
		 epochLoss_class/i, time() - TIME)


		#generate samples after each 10 epochs
		if e % 1 == 0:
			cvae.eval()
			dis.eval()

			#Load test data
			testIter = iter(testLoader)
			xTest, yTest = testIter.next()
			yTest = yTest
			if cvae.useCUDA:
				xTest = Variable(xTest).cuda().data
				yTest = Variable(yTest).cuda()
				outputs, outMu, outLogVar, outY = cvae(Variable(xTest)) 
			else:
				yTest = Variable(yTest)

			print 'saving a set of samples'
			if cvae.useCUDA:
				z = Variable(torch.randn(xTest.size(0), opts.nz).cuda())
			else:
				z = Variable(torch.randn(xTest.size(0), opts.nz))


			ySmile = Variable(torch.eye(2)[torch.LongTensor(np.ones(yTest.size()).astype(int))]).type_as(z)
			samples = cvae.decode(ySmile, z).cpu()
			save_image(samples.data, join(exDir,'smile_epoch'+str(e)+'.png'))

			yNoSmile = Variable(torch.eye(2)[torch.LongTensor(np.zeros(yTest.size()).astype(int))]).type_as(z)
			samples = cvae.decode(yNoSmile, z).cpu()
			save_image(samples.data, join(exDir,'no_smile_epoch'+str(e)+'.png'))

			#check reconstructions after each 10 epochs
			outputs, outMu, outLogVar, outY = cvae(Variable(xTest))

			bceLossTest, klLossTest = vae_loss_fn(rec_x=outputs.data, x=xTest, mu=outMu, logVar=outLogVar)
			maxVal, predLabel = torch.max(outY, 1)
			classScoreTest = torch.eq(predLabel, yTest).float().sum()/yTest.size(0)
			print 'classification test:', classScoreTest.data[0]

			save_image(xTest, join(exDir,'input.png'))
			save_image(outputs.data, join(exDir,'output_'+str(e)+'.png'))

			label_switch(xTest, yTest, cvae, exDir=exDir)

			cvae.save_params(exDir=exDir)

			losses['total'].append(epochLoss/Ns)
			losses['kl'].append(epochLoss_kl/Ns)
			losses['bce'].append(epochLoss_bce/Ns)
			losses['test_bce'].append((bceLossTest).data[0]/xTest.size(0)) #append every epoch
			losses['dis'].append(epochLoss_dis/Ns)
			losses['gen'].append(epochLoss_gen/Ns)
			losses['class'].append(epochLoss_class/Ns)
			losses['test_class'].append(classScoreTest.data[0])

			if e > 1:
				plot_losses(losses, exDir, epochs=e+1)
				plot_norm_losses(losses, exDir, epochs=1+e)




