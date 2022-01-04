import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock



class Enhancer(nn.Module):
	def __init__(self, dim=256, n_blocks=15, cf=5120):
		super(Enhancer, self).__init__()
		self.dim = dim
		self.cf = cf # prenet output c * f (64 * 51)
		self.n_blocks = n_blocks
		self.prenet = nn.Sequential(
			nn.Conv2d(1, self.dim//8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.dim//8, self.dim//8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.dim//8, self.dim//4, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
		)
		self.conformer_block = nn.ModuleList()
		for i in range(self.n_blocks):
			self.conformer_block.append(ConformerBlock(dim=self.dim, dim_head=64, heads=4, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=9, attn_dropout=0.1))
		self.proj_in = nn.Conv1d(self.cf, self.dim, 1, 1)
		self.proj_out = nn.Conv1d(self.dim, 80, 1, 1)
	def forward(self, x):
		# x [b, f, t]
		x = self.prenet(x)
		b, c, f, t = x.size()
		x = x.permute(0,3,1,2) # [b, t, c, f]
		x = x.contiguous().view(b, t, c*f) # [b, t, f]
		x = x.permute(0,2,1) # [b, f, t]
		x = self.proj_in(x)
		x = x.permute(0,2,1)
		for layer in self.conformer_block:
			x = layer(x)
		x = x.permute(0,2,1)
		mask = self.proj_out(x)
		return mask.unsqueeze(1)


class Classifier(nn.Module):
	def __init__(self, dim=256, n_blocks=15, cf=640):
		super(Classifier, self).__init__()
		self.dim = dim
		self.cf = cf # prenet output c * f (64 * 51)
		self.n_blocks = n_blocks
		self.prenet = nn.Sequential(
			nn.Conv2d(2, self.dim//8, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.dim//8, self.dim//8, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.dim//8, self.dim//4, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
		)
		self.conformer_block = nn.ModuleList()
		for i in range(self.n_blocks):
			self.conformer_block.append(ConformerBlock(dim=self.dim, dim_head=64, heads=4, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=9, attn_dropout=0.1))
		self.proj_in = nn.Conv1d(self.cf, self.dim, 1, 1)
		self.proj_out = nn.Conv1d(self.dim, 3, 1, 1)
	def forward(self, x):
		# x [b, f, t]
		x = self.prenet(x)
		b, c, f, t = x.size()
		x = x.permute(0,3,1,2) # [b, t, c, f]
		x = x.contiguous().view(b, t, c*f) # [b, t, f]
		x = x.permute(0,2,1) # [b, f, t]
		x = self.proj_in(x)
		x = x.permute(0,2,1)
		for layer in self.conformer_block:
			x = layer(x)
		x = x.permute(0,2,1)
		logits = self.proj_out(x)
		return logits






class model(nn.Module):
	def __init__(self):
		super(model, self).__init__()
		self.enhancer = Enhancer()
		self.classifier = Classifier()
		
	def forward(self, x):
		x_log = torch.log(x.clamp(min=1e-5))
		mask = self.enhancer(x_log)
		# mask = torch.sigmoid(mask)
		logit = self.classifier(torch.cat((x,mask),axis=1))
		return logit, mask

