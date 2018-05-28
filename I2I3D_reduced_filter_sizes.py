import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


### IMPORTANT INFO FOR USAGE ###
# The model has three (avg) pooling layers. This requires the input to the network to be
# a multiple of 8 (input size * (1/2)^3 needs to be an integer). Either preprocess image
# to fit this requirement or extend the forward function below to do it for you. As of now,
# code raises error to notify. 

n = 4										# 4 - 8 - 16 - 32 filter sizes

class i2i(nn.Module):

	def __init__(self):
		super(i2i, self).__init__()
		
		# Pooling layer, avg pooling
		self.pool = nn.AvgPool3d(2)

		# Layers going down
		# Number in name represents deepness level in network
		self.d1 = nn.Sequential(
			nn.Conv3d(1, n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(n, n, 3, padding=1),
			nn.ReLU(True)
			)
		self.d2 = nn.Sequential(
			nn.Conv3d(n, 2*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(2*n, 2*n, 3, padding=1),
			nn.ReLU(True),
			)
		self.d3 = nn.Sequential(
			nn.Conv3d(2*n, 4*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(4*n, 4*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(4*n, 4*n, 3, padding=1),
			nn.ReLU(True)
			)
		self.d4 = nn.Sequential(
			nn.Conv3d(4*n, 8*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(8*n, 8*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(8*n, 8*n, 3, padding=1),
			nn.ReLU(True)
			)

		# Upscaling pre-merge Layers
		self.merge3_up = nn.ConvTranspose3d(8*n, 8*n, 4, stride=2, groups=8*n)
		self.merge2_up = nn.ConvTranspose3d(4*n, 4*n, 4, stride=2, groups=4*n)
		self.merge1_up = nn.ConvTranspose3d(2*n, 2*n, 4, stride=2, groups=2*n)

		# Layers going up
		self.u3 = nn.Sequential(
			nn.Conv3d(8*n+4*n, 4*n, 1),
			nn.ReLU(True),
			nn.Conv3d(4*n, 4*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(4*n, 4*n, 3, padding=1),
			nn.ReLU(True)
			)
		self.u2 = nn.Sequential(
			nn.Conv3d(4*n+2*n, 2*n, 1),
			nn.ReLU(True),
			nn.Conv3d(2*n, 2*n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(2*n, 2*n, 3, padding=1),
			nn.ReLU(True)
			)
		self.u1 = nn.Sequential(
			nn.Conv3d(2*n+n, n, 1),
			nn.ReLU(True),
			nn.Conv3d(n, n, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(n, n, 3, padding=1),
			nn.ReLU(True)
			)
		self.s1 = nn.Conv3d(n, 1, 1)

		# Upscaling score layers
		self.s2_up = nn.Sequential(
			nn.Conv3d(2*n, 1, 1),
			nn.ConvTranspose3d(1, 1, 4, stride=2, groups=1)
			)
		self.s3_up = nn.Sequential(
			nn.Conv3d(4*n, 1, 1),
			nn.ConvTranspose3d(1, 1, 8, stride=4, groups=1)
			)
		# self.s4_up = nn.Sequential(
		# 	nn.Conv3d(8*n, 1, 1),
		# 	nn.ConvTranspose3d(1, 1, 16, stride=8, groups=1)
		# 	)


	def forward(self, x):

		# Check if divisible by 8 first.
		shape = x.shape
		if shape[2]%8!=0 or shape[3]%8!=0 or shape[4]%8!=0:
			raise ValueError('Size of input image needs to be divisible by 8:', shape)
			
		d1 = self.d1(x)
		d2 = self.d2(self.pool(d1))
		d3 = self.d3(self.pool(d2))
		d4 = self.d4(self.pool(d3))
		
		# Score lvl 4
		#s4 = F.pad( self.s4_up(d4) , (-4,-4,-4,-4,-4,-4))

		# Merge (upscale + concatenate) and ascend
		u3 = self.u3( torch.cat(( F.pad( self.merge3_up(d4), (-1,-1,-1,-1,-1,-1)) , d3),1))
		del d4, d3

		# Score lvl 3
		s3 = F.pad( self.s3_up(u3) , (-2,-2,-2,-2,-2,-2))

		# Merge (upscale + concatenate) and ascend
		u2 = self.u2( torch.cat(( F.pad( self.merge2_up(u3), (-1,-1,-1,-1,-1,-1)) , d2),1))
		del u3, d2

		# Score lvl 2
		s2 = F.pad( self.s2_up(u2) , (-1,-1,-1,-1,-1,-1))

		# Merge (upscale + concatenate) and ascend
		u1 = self.u1( torch.cat(( F.pad( self.merge1_up(u2), (-1,-1,-1,-1,-1,-1)) , d1),1))
		del u2, d1

		# Score lvl 1
		s1 = self.s1(u1)

		# Sigmoid could be added here, but was added outside of network
		# in this thesis.
		return s1, s2, s3#, s4