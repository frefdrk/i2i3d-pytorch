import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


### IMPORTANT INFO FOR USAGE ###
# The model has three (avg) pooling layers. This requires the input to the network to be
# a multiple of 8 (input size * (1/2)^3 needs to be an integer). Either preprocess image
# to fit this requirement or extend the forward function below to do it for you. As of now,
# code raises error to notify. 

class i2i(nn.Module):

	def __init__(self):
		super(i2i, self).__init__()
		
		# Pooling layer, avg pooling
		self.pool = nn.AvgPool3d(2)

		# Layers going down
		# Number in name represents deepness level in network
		self.encoder_lvl_1 = nn.Sequential(
			nn.Conv3d(1, 32, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(32, 32, 3, padding=1),
			nn.ReLU(True)
			)
		self.encoder_lvl_2 = nn.Sequential(
			nn.Conv3d(32, 128, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(128, 128, 3, padding=1),
			nn.ReLU(True),
			)
		self.encoder_lvl_3 = nn.Sequential(
			nn.Conv3d(128, 256, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(256, 256, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(256, 256, 3, padding=1),
			nn.ReLU(True)
			)
		self.encoder_lvl_4 = nn.Sequential(
			nn.Conv3d(256, 512, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(512, 512, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(512, 512, 3, padding=1),
			nn.ReLU(True)
			)

		# Upscaling pre-merge Layers
		self.merge3_up = nn.ConvTranspose3d(512, 512, 4, stride=2, groups=512)
		self.merge2_up = nn.ConvTranspose3d(256, 256, 4, stride=2, groups=256)
		self.merge1_up = nn.ConvTranspose3d(128, 128, 4, stride=2, groups=128)

		# Layers going up
		self.decoder_lvl_3 = nn.Sequential(
			nn.Conv3d(768, 256, 1),
			nn.ReLU(True),
			nn.Conv3d(256, 256, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(256, 256, 3, padding=1),
			nn.ReLU(True)
			)
		self.decoder_lvl_2 = nn.Sequential(
			nn.Conv3d(384, 128, 1),
			nn.ReLU(True),
			nn.Conv3d(128, 128, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(128, 128, 3, padding=1),
			nn.ReLU(True)
			)
		self.decoder_lvl_1 = nn.Sequential(
			nn.Conv3d(160, 32, 1),
			nn.ReLU(True),
			nn.Conv3d(32, 32, 3, padding=1),
			nn.ReLU(True),
			nn.Conv3d(32, 32, 3, padding=1),
			nn.ReLU(True)
			)
		self.s1 = nn.Conv3d(32, 1, 1)

		# Upscaling score layers
		self.s2_up = nn.Sequential(
			nn.Conv3d(128, 1, 1),
			nn.ConvTranspose3d(1, 1, 4, stride=2, groups=1)
			)
		self.s3_up = nn.Sequential(
			nn.Conv3d(256, 1, 1),
			nn.ConvTranspose3d(1, 1, 8, stride=4, groups=1)
			)
		self.s4_up = nn.Sequential(
			nn.Conv3d(512, 1, 1),
			nn.ConvTranspose3d(1, 1, 16, stride=8, groups=1)
			)


	def forward(self, x):

		# Check if divisible by 8 first.
		shape = x.shape
		if shape[2]%8!=0 or shape[3]%8!=0 or shape[4]%8!=0:
			raise ValueError('Size of input image is not divisible by 8:', shape)

		# d for moving "down" the encoder 
		d1 = self.encoder_lvl_1(x)
		d2 = self.encoder_lvl_2(self.pool(d1))
		d3 = self.encoder_lvl_3(self.pool(d2))
		d4 = self.encoder_lvl_4(self.pool(d3))

		# Score lvl 4
		s4 = F.pad( self.s4_up(d4) , (-4,-4,-4,-4,-4,-4))

		# u for moving "up" the decoder
		# Merge (upscale + concatenate) and ascend
		u3 = self.decoder_lvl_3( torch.cat(( F.pad( self.merge3_up(d4), (-1,-1,-1,-1,-1,-1)) , d3),1))
		del d4, d3

		# Score lvl 3
		s3 = F.pad( self.s3_up(u3) , (-2,-2,-2,-2,-2,-2))

		# Merge (upscale + concatenate) and ascend
		u2 = self.decoder_lvl_2( torch.cat(( F.pad( self.merge2_up(u3), (-1,-1,-1,-1,-1,-1)) , d2),1))
		del u3, d2

		# Score lvl 2
		s2 = F.pad( self.s2_up(u2) , (-1,-1,-1,-1,-1,-1))

		# Merge (upscale + concatenate) and ascend
		u1 = self.decoder_lvl_1( torch.cat(( F.pad( self.merge1_up(u2), (-1,-1,-1,-1,-1,-1)) , d1),1))
		del u2, d1
		
		# Score lvl 1
		s1 = self.s1(u1)

		return s1, s2, s3, s4