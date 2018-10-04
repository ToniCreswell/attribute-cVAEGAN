# Conditional Autoencoders with Adversarial Information Factorization

(New version of the code using ResNets will be provided soon to reproduce results in the latest verision of our paper: https://arxiv.org/pdf/1711.05175.pdf)

## Updates to our paper and code (coming soon) include:
1. Classification results on CelebA facial attributes
2. Use of ResNets to improve image quality

## To use code:
1. Download the celebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Install dependancies listed in requirements.txt
3. You will also need pyTorch which may be downloaded from http://pytorch.org
4. Run the jupyter notebooks to get the data tensors xTrain.npy and yAllTrain.npy and move them in to folder celebA/InData/
5. The code may be run from cmd line with various options detailed in the code


## Example results:

![alt text](https://github.com/ToniCreswell/attribute-cVAEGAN/blob/master/Experiments/rec_0.png)
![alt text](https://github.com/ToniCreswell/attribute-cVAEGAN/blob/master/Experiments/rec_1.png)