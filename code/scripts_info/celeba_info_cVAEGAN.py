#conditional VAE+GAN trained on smile/no smile faces -- the improved version!
import sys
sys.path.append('../')

from dataload import CELEBA
from function import make_new_folder, plot_losses, vae_loss_fn, save_input_args, \
is_ready_to_stop_pretraining, sample_z, class_loss_fn, plot_norm_losses,\
binary_class_score #, one_hot
from function import label_switch_1 as label_switch
from function import soft_label_switch_1 as soft_label_switch
from models import CVAE1, DISCRIMINATOR, AUX
from models import DISCRIMINATOR as CLASSIFIER

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
	parser.add_argument('--lr', default=2e-4, type=float)
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	parser.add_argument('--outDir', default='../../Experiments/celebA_info_cVAEGAN', type=str)
	parser.add_argument('--commit', default='None', type=str)
	parser.add_argument('--alpha', default=1, type=float) #weight on the KL divergance
	parser.add_argument('--gamma', default=1, type=float) #weight on the aux enc loss
	parser.add_argument('--delta', default=1, type=float) #weight on the adversarial loss
	parser.add_argument('--rho', default=1, type=float) #weight on the class loss for the vae
	parser.add_argument('--comments', type=str)
	parser.add_argument('--mom', default=0.9, type=float) #momentum in rms prop of dis
	parser.add_argument('--weightDecay', default=0, type=float) #weight decay in RMSprop for all params
	parser.add_argument('--load_VAE_from', default=None, type=str)
	parser.add_argument('--load_CLASSER_from', default='../../Experiments_delta_z/celeba_joint_VAE_DZ/Ex_15', type=str)
	parser.add_argument('--evalMode', action='store_true')
	parser.add_argument('--beta', default=1, type=float)  #weight on the rec class loss to update VAE
	parser.add_argument('--label', default='Smiling', type=str)

	return parser.parse_args()

def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size(0),1).type_as(x)
	return x,y


def evaluate(cvae, testLoader, exDir, e=1, classifier=None):  #e is the epoch

	cvae.eval()

	#Load test data
	xTest, yTest = prep_data(iter(testLoader).next(), cvae.useCUDA)
	

	print 'saving a set of samples'
	if cvae.useCUDA:
		z = Variable(torch.randn(xTest.size(0), opts.nz).cuda())
	else:
		z = Variable(torch.randn(xTest.size(0), opts.nz))

	ySmile = Variable(torch.Tensor(yTest.size()).fill_(1)).type_as(yTest)
	samples = cvae.decode(ySmile, z).cpu()
	save_image(samples.data, join(exDir,'smile_epoch'+str(e)+'.png'))

	yNoSmile = Variable(torch.Tensor(yTest.size()).fill_(0)).type_as(yTest)
	samples = cvae.decode(yNoSmile, z).cpu()
	save_image(samples.data, join(exDir,'no_smile_epoch'+str(e)+'.png'))

	#check reconstructions after each 10 epochs
	outputs, outMu, outLogVar, outY = cvae(xTest)


	bceLossTest, klLossTest = cvae.loss(rec_x=outputs, x=xTest, mu=outMu, logVar=outLogVar)
	predLabel = torch.floor(outY)

	classScoreTest= binary_class_score(predLabel, yTest, thresh=0.5)
	print 'classification test:', classScoreTest.data[0]

	save_image(xTest.data, join(exDir,'input.png'))
	save_image(outputs.data, join(exDir,'output_'+str(e)+'.png'))

	rec1, rec0 = label_switch(xTest.data, yTest, cvae, exDir=exDir)

	# for further eval
	if e == 'evalMode' and classer is not None:
		classer.eval()
		yPred0 = classer(rec0)
		y0 = Variable(torch.LongTensor(yPred0.size()).fill_(0)).type_as(yTest)
		class0 = binary_class_score(yPred0, y0, thresh=0.5)
		yPred1 = classer(rec1)
		y1 = Variable(torch.LongTensor(yPred1.size()).fill_(1)).type_as(yTest)
		class1 = binary_class_score(yPred1, y1, thresh=0.5)

		f = open(join(exDir, 'eval.txt'), 'w')
		f.write('Test MSE:'+ str(F.mse_loss(outputs, xTest).data[0]))
		f.write('Class0:'+ str(class0.data[0]))
		f.write('Class1:'+ str(class1.data[0]))
		f.close()


	return (bceLossTest).data[0]/xTest.size(0), classScoreTest.data[0]

if __name__=='__main__':
	opts = get_args()

	####### Data set #######
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
	trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor(), label=opts.label)
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor(), label=opts.label)
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'


	####### Create model #######
	cvae = CVAE1(nz=opts.nz, imSize=64, fSize=opts.fSize)
	dis = DISCRIMINATOR(imSize=64, fSize=opts.fSize) 
	aux = AUX(nz=opts.nz)
	classer = CLASSIFIER(imSize=64, fSize=64) #for eval only! 

	if cvae.useCUDA:
		print 'using CUDA'
		cvae.cuda()
		dis.cuda()
		aux.cuda()
		classer.cuda()
	else: print '\n *** NOT USING CUDA ***\n'

	#load model is applicable
	if opts.load_VAE_from is not None:
		cvae.load_params(opts.load_VAE_from)

	if opts.evalMode:
		classer.load_params(opts.load_CLASSER_from)

		assert opts.load_VAE_from is not None
		#make a new folder to save eval results w/out affecting others
		evalDir=join(opts.load_VAE_from,'evalFolder')
		print 'Eval results will be saved to', evalDir
		try:  #may already have an eval folder
			os.mkdir(evalDir)
		except:
			print 'file already created'
		_, _ = evaluate(cvae, testLoader, evalDir, e='evalMode', classifier=classer)
		exit()


	print cvae
	print dis
	print aux

	####### Define optimizer #######
	optimizerCVAE = optim.RMSprop(cvae.parameters(), lr=opts.lr, weight_decay=opts.weightDecay)  #specify the params that are being upated
	optimizerDIS = optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.mom, weight_decay=opts.weightDecay)
	optimizerAUX = optim.RMSprop(aux.parameters(), lr=opts.lr, weight_decay=opts.weightDecay)

	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)  #save training opts


	losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[], 'aux':[], 'auxEnc':[]}
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
		epochLoss_aux = 0
		epochLoss_auxEnc = 0

		TIME = time()

		for i, data in enumerate(trainLoader, 0):
			
			x, y = prep_data(data, cvae.useCUDA)

			#get ouput, clac loss, calc all grads, optimise
			#VAE loss
			outRec, outMu, outLogVar, predY = cvae(x)
			z = cvae.re_param(outMu, outLogVar)
			bceLoss, klLoss = cvae.loss(rec_x=outRec, x=x, mu=outMu, logVar=outLogVar)
			vaeLoss = bceLoss + opts.alpha * klLoss

			#Classification loss #not on reconstructed sample
			classLoss = F.binary_cross_entropy(predY.type_as(y), y)
			vaeLoss += opts.rho * classLoss

			#Class loss on reconstruction
			_, _, yRec = cvae.encode(outRec)
			classRecLoss = F.binary_cross_entropy(yRec, y)
			vaeLoss += opts.beta * classRecLoss

			#Train the encoder to NOT predict y from z
			auxY = aux(z)  #not detached update the encoder!
			auxEncLoss = F.binary_cross_entropy(auxY.type_as(y), y)  
			vaeLoss -= opts.gamma * auxEncLoss

			#Train the aux net to predict y from z
			auxY = aux(z.detach())  #detach: to ONLY update the AUX net #the prediction here for GT being predY
			auxLoss = F.binary_cross_entropy(auxY.type_as(y), y) #correct order  #predY is a Nx2 use 2nd col.
			
			#DIS loss
			pXreal = dis(x)
			pXfakeRec = dis(outRec.detach())
			zRand = sample_z(x.size(0), opts.nz, cvae.useCUDA)
			yRand = y
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

			optimizerAUX.zero_grad()
			auxLoss.backward()
			optimizerAUX.step()

			optimizerDIS.zero_grad()
			disLoss.backward()
			optimizerDIS.step()

			epochLoss += vaeLoss.data[0]
			epochLoss_kl += klLoss.data[0]
			epochLoss_bce += bceLoss.data[0]
			epochLoss_gen += genLoss.data[0]
			epochLoss_dis += disLoss.data[0]
			epochLoss_class += classLoss.data[0]
			epochLoss_aux += auxLoss.data[0]
			epochLoss_auxEnc += auxEncLoss.data[0]

			if i%100==1:
				i+=1
				print '[%d, %d] loss: %0.5f, gen: %0.5f, dis: %0.5f, bce: %0.5f, kl: %0.5f, aux: %0.5f, time: %0.3f' % \
		 			(e, i, epochLoss/i, epochLoss_dis/i, epochLoss_gen/i, epochLoss_bce/i, epochLoss_kl/i, epochLoss_aux/i, time() - TIME)

	
		#generate samples after each 10 epochs

		normbceLossTest, classScoreTest = evaluate(cvae, testLoader, exDir, e=e)

		cvae.save_params(exDir=exDir)


		losses['total'].append(epochLoss/Ns)
		losses['kl'].append(epochLoss_kl/Ns)
		losses['bce'].append(epochLoss_bce/Ns)
		losses['test_bce'].append(normbceLossTest) #append every epoch
		losses['dis'].append(epochLoss_dis/Ns)
		losses['gen'].append(epochLoss_gen/Ns)
		losses['class'].append(epochLoss_class/Ns)
		losses['test_class'].append(classScoreTest)
		losses['aux'].append(epochLoss_aux/Ns)
		losses['auxEnc'].append(epochLoss_auxEnc/Ns)

		if e > 1:
			plot_losses(losses, exDir, epochs=e+1)
			plot_norm_losses(losses, exDir, epochs=e+1)

	normbceLossTest, classScoreTest = evaluate(cvae, testLoader, exDir, e='evalMode')


