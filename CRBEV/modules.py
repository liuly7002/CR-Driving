from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class TransformModule(nn.Module):
	def __init__(self, dim=(37, 60), num_view=1):
		super(TransformModule, self).__init__()
		self.num_view = num_view
		self.dim = dim
		self.mat_list = nn.ModuleList()

		for i in range(self.num_view):
			fc_transform = nn.Sequential(
						nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
						nn.ReLU(),
						nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
						nn.ReLU()
					)
			self.mat_list += [fc_transform]

	def forward(self, x):
		x = x.view(list(x.size()[:3]) + [self.dim[0] * self.dim[1],])
		view_comb = self.mat_list[0](x[:, 0])
		for index in range(x.size(1))[1:]:
			view_comb += self.mat_list[index](x[:, index])
		view_comb = view_comb.view(list(view_comb.size()[:2]) + list(self.dim))
		return view_comb


class Graph_Pred(nn.Module):
	def __init__(self, num_views, num_class, output_size, map_extents, map_resolution):
		super(Graph_Pred, self).__init__()
		self.num_views = num_views
		self.output_size = output_size

		self.seg_size = (
			int((map_extents[3] - map_extents[1]) / map_resolution),
			int((map_extents[2] - map_extents[0]) / map_resolution),
		)

		self.transform_module = TransformModule(dim=self.output_size, num_view=self.num_views)
		self.decoder = EnhancedGraph(num_class)

	def forward(self, x, lambda_t, *args):
		B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] + list(x.size()[2:])).size()
		
		x = x.view(B*N, C, H, W)
		x = x.view([B, N] + list(x.size()[1:]))
		x = self.transform_module(x)
		
		x = self.decoder(x)

		# CAPM

		return x

class EnhancedGraph(nn.Module):
	def __init__(self, num_class=6):
		super(EnhancedGraph, self).__init__()

		self.class_weights = nn.Parameter(torch.ones(num_class))

		self.main_branch = nn.Sequential(
			nn.AdaptiveAvgPool2d((18, 18)),
			nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
			nn.GroupNorm(16, 256),
			nn.ReLU(inplace=True)
		)

		self.scale_convs = nn.ModuleList([
			nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
			nn.Conv2d(256, 64, kernel_size=3, padding=2, dilation=2, bias=False),
			nn.Conv2d(256, 64, kernel_size=3, padding=4, dilation=4, bias=False),
			nn.Conv2d(256, 64, kernel_size=3, padding=8, dilation=8, bias=False)
		])
		
		self.scale_norms = nn.ModuleList([
			nn.GroupNorm(4, 64) for _ in range(4)
		])

		self.spatial_attention = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=7, padding=3),
			nn.GroupNorm(4, 64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 1, kernel_size=1),
			nn.Sigmoid()
		)

		self.channel_attention = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(256, 64, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 256, 1),
			nn.Sigmoid()
		)

		self.front_enhance = nn.ModuleDict({
			'main': nn.Sequential(
				nn.Conv2d(256, 256, kernel_size=3, padding=1),
				nn.GroupNorm(16, 256),
				nn.ReLU(inplace=True),
				nn.Conv2d(256, 256, kernel_size=3, padding=1),
				nn.GroupNorm(16, 256),
				nn.ReLU(inplace=True)
			),
			'large_field': nn.Sequential(
				nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
				nn.GroupNorm(8, 128),
				nn.ReLU(inplace=True),
				nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
				nn.GroupNorm(8, 128),
				nn.ReLU(inplace=True)
			),
			'attention': nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(384, 96, 1),
				nn.ReLU(inplace=True),
				nn.Conv2d(96, 384, 1),
				nn.Sigmoid()
			),
			'fusion': nn.Sequential(
				nn.Conv2d(384, 256, 1),
				nn.GroupNorm(16, 256),
				nn.ReLU(inplace=True)
			)
		})

		self.classifier = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.GroupNorm(16, 256),
			nn.ReLU(inplace=True),
			nn.Dropout2d(0.1),
			nn.Conv2d(256, num_class, kernel_size=1)
		)
        
	def forward(self, x):

		main_feat = self.main_branch(x)

		scale_feats = []
		for conv, norm in zip(self.scale_convs, self.scale_norms):
			feat = conv(main_feat)
			feat = norm(feat)
			feat = F.relu(feat, inplace=True)
			scale_feats.append(feat)

		multi_scale_feat = torch.cat(scale_feats, dim=1)

		spatial_weight = self.spatial_attention(main_feat)

		channel_weight = self.channel_attention(multi_scale_feat)

		enhanced_feat = multi_scale_feat * channel_weight
		enhanced_feat = enhanced_feat * spatial_weight

		B, C, H, W = enhanced_feat.shape
		front_region = enhanced_feat[:, :, :H//3, :]

		main_front = self.front_enhance['main'](front_region)
		large_front = self.front_enhance['large_field'](front_region)

		front_cat = torch.cat([main_front, large_front], dim=1)

		front_attention = self.front_enhance['attention'](front_cat)
		front_weighted = front_cat * front_attention

		front_enhanced = self.front_enhance['fusion'](front_weighted)

		front_enhanced = front_enhanced + front_region

		enhanced_feat[:, :, :H//3, :] = front_enhanced

		out = self.classifier(enhanced_feat)

		out = out * self.class_weights.view(1, -1, 1, 1)
		
		return out