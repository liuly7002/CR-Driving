import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from CRBEV.model import CRPlan
from CRBEV.data import CR_Data
from CRBEV.config import GlobalConfig
from CRBEV.metric import Gragh_Metric, Cal_IoU

from CRBEV.graph_loss import CombinedLoss

import math

import logging
logger = logging.getLogger(__name__)
class CR_planner(pl.LightningModule):
	def __init__(self, config, lr, checkpoint_path=None):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = CRPlan(config)
		self._load_weight()
		self.load_ckpt()
		self.val_counter = -1

		if checkpoint_path is not None:
			try:
				checkpoint = torch.load(checkpoint_path, map_location='cpu')
				self.load_state_dict(checkpoint['state_dict'])
				print(f"Successfully loaded model from checkpoint: {checkpoint_path}")
				if 'val_counter' in checkpoint:
					self.val_counter = checkpoint['val_counter']
				else:
					import re
					match = re.search(r'epoch=(\d+)', checkpoint_path)
					if match:
						self.val_counter = int(match.group(1))
			except Exception as e:
				print(f"Error loading checkpoint: {e}")

	def _load_weight(self):
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)

	def load_ckpt(self):
		# load the pre-train weight(TCP)
		ckpt = torch.load('./pretrain_weight.ckpt')
		ckpt = ckpt["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.", "")
			new_state_dict[new_key] = value
		self.model.load_state_dict(new_state_dict, strict=False)
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		front_img = batch['front_img']

		speed = batch['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']

		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1, 1)
		feature = batch['feature']

		gt_waypoints = batch['waypoints']
		gt_graph = batch['risk_map_18x18_6'][:, 0, :, :]

		lambda_t = 1

		pred = self.model(front_img,state,target_point,lambda_t)

		class_weights = [3.0, 7.8, 8.2, 8.0, 5.2, 5.5]
		class_weights = torch.FloatTensor(class_weights).cuda()
		alpha = [0.7, 1.6, 1.8, 1.7, 1.1, 1.1]
		alpha = torch.FloatTensor(alpha).cuda()
		loss_fn = CombinedLoss(alpha=alpha, gamma=2.7, weight=class_weights, smooth=0.1)
		graph_loss = torch.abs(loss_fn(pred['graph'], gt_graph) * self.config.graph_weight)

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5

		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight

		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight

		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'], feature)) * self.config.features_weight

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len

		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

		loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss + graph_loss

		self.log('train_graph_loss', graph_loss.item())
		self.log('train_action_loss', action_loss.item())
		self.log('train_speed_loss', speed_loss.item())
		self.log('train_value_loss', value_loss.item())
		self.log('train_feature_loss', feature_loss.item())
		self.log('train_future_feature_loss', future_feature_loss.item())
		self.log('train_future_action_loss', future_action_loss.item())
		self.log('train_wp_loss_loss', wp_loss.item())
		self.log('train_loss', loss.item())

		return loss

	def configure_optimizers(self):
		optimizer = optim.AdamW(
			self.parameters(), 
			lr=self.lr, 
			weight_decay=5e-4,
			betas=(0.9, 0.999)
		)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer, 
			T_0=20,
			T_mult=2,
			eta_min=self.lr * 0.01
		)
		
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		front_img = batch['front_img']

		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)

		value = batch['value'].view(-1, 1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']

		gt_graph = batch['risk_map_18x18_6'][:, 0, :, :]

		lambda_t = 1

		pred = self.model(front_img, state, target_point, lambda_t)

		class_weights = [3.0, 7.8, 8.2, 8.0, 5.2, 5.5]
		class_weights = torch.FloatTensor(class_weights).cuda()
		alpha = [0.7, 1.6, 1.8, 1.7, 1.1, 1.1]
		alpha = torch.FloatTensor(alpha).cuda()
		loss_fn = CombinedLoss(alpha=alpha, gamma=2, weight=class_weights, smooth=0.1)
		graph_loss = torch.abs(loss_fn(pred['graph'], gt_graph) * self.config.graph_weight)

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5

		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight

		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight

		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature)) * self.config.features_weight

		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

		B = batch['action_mu'].shape[0]
		batch_steer_l1 = 0 
		batch_brake_l1 = 0
		batch_throttle_l1 = 0
		for i in range(B):
			throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
			batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
			batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
			batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

		batch_throttle_l1 /= B
		batch_steer_l1 /= B
		batch_brake_l1 /= B

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len-1):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len

		val_loss = wp_loss + batch_throttle_l1 + 5*batch_steer_l1 + batch_brake_l1 + graph_loss

		self.log("val_graph_loss", graph_loss.item(), sync_dist=True)
		self.log("val_action_loss", action_loss.item(), sync_dist=True)
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
		self.log('val_value_loss', value_loss.item(), sync_dist=True)
		self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
		self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
		self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
		self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
		self.log('val_loss', val_loss.item(), sync_dist=True)

		return action_loss.item(), speed_loss.item(), value_loss.item(), feature_loss.item(), \
			wp_loss.item(), future_feature_loss.item(), future_action_loss.item(), val_loss.item(), \
			graph_loss.item(), gt_graph, pred['graph']

	def validation_epoch_end(self, outputs):
		total_action_loss = 0
		total_speed_loss = 0
		total_value_loss = 0
		total_feature_loss = 0
		total_wp_loss = 0
		total_future_feature_loss = 0
		total_future_action_loss = 0
		total_val_loss = 0

		total_graph_loss = 0
		total_gt_graph = []
		total_pred_graph = []

		for action_loss, speed_loss, value_loss, feature_loss, wp_loss, future_feature_loss,\
				future_action_loss, val_loss, graph_loss, gt_graph, pred_graph in outputs:
			total_action_loss += action_loss
			total_speed_loss += speed_loss
			total_value_loss += value_loss
			total_feature_loss += feature_loss
			total_wp_loss += wp_loss
			total_future_feature_loss += future_feature_loss
			total_future_action_loss += future_action_loss
			total_val_loss += val_loss
			total_graph_loss += graph_loss
			total_gt_graph.append(gt_graph)
			total_pred_graph.append(pred_graph)

		f1_graph_overall, f1_graph_cate, f1_graph_mean = Gragh_Metric(total_gt_graph, total_pred_graph)
		iou_graph = Cal_IoU(total_gt_graph, total_pred_graph)

		val_info = """
		epoch {0}
		-----------------------
		val_total_loss: {1}
		val_graph_loss: {2}
		
		graph_category: \n{3}
		graph_overall: {4}
		graph_mean: {5}
		\ngraph_iou: \n{6}
		
				""".format(self.val_counter,
							total_val_loss, total_graph_loss,
							f1_graph_cate, f1_graph_overall, f1_graph_mean, iou_graph
							)
		print(val_info)
		result_file = './log.txt'
		with open(result_file, 'a') as f:
			f.write(val_info)
		self.val_counter += 2


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=str, default='weight_file', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=30, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=2, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=4, help='number of gpus')

	parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='Gradient accumulation steps')
	parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
	parser.add_argument('--resume_epoch', type=int, default=None, help='Resume training from this epoch')
	parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor to reduce learning rate when resuming training')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	config = GlobalConfig()

	train_set = CR_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug)
	val_set = CR_Data(root=config.root_dir_all, data_folders=config.val_data,)
	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	if args.checkpoint is not None and args.lr_factor < 1.0:
		args.lr = args.lr * args.lr_factor
		print(f"Resuming training with reduced learning rate: {args.lr}")

	model = CR_planner(config, args.lr, args.checkpoint)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=-1, save_last=True,
											dirpath=args.logdir, filename="{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

	trainer_kwargs = {
		'default_root_dir': args.logdir,
		'gpus': args.gpus,
		'accelerator': 'ddp',
		'sync_batchnorm': True,
		'plugins': DDPPlugin(find_unused_parameters=False),
		'profiler': 'simple',
		'benchmark': True,
		'log_every_n_steps': 1,
		'flush_logs_every_n_steps': 5,
		'callbacks': [checkpoint_callback],
		'check_val_every_n_epoch': args.val_every,
		'max_epochs': args.epochs,
		'accumulate_grad_batches': args.accumulate_grad_batches,
		'gradient_clip_val': 1.0
	}

	if args.checkpoint is not None:
		trainer_kwargs['resume_from_checkpoint'] = args.checkpoint

	trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)

	trainer.fit(model, dataloader_train, dataloader_val)








