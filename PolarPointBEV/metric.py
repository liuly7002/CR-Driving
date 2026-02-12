from sklearn.metrics import f1_score, classification_report
import torch
import numpy as np
import json


def List2List(List):
	Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
	Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])
	Arr = np.vstack((Arr1, Arr2))

	return [i for item in Arr for i in item]

def Gragh_Metric(total_gt_graph, total_pred_graph):
	graph_num = len(total_gt_graph)
	gt_graph_list = []    # 所有图像的真实标签
	pred_graph_list = []  # 所有图像的预测标签

	for idx in range(graph_num):
		gt_graph = total_gt_graph[idx].cpu().numpy()   # 将真实标签的图像从 GPU 转移到 CPU，并转换为 NumPy 数组。
		pred_graph = torch.argmax(total_pred_graph[idx], dim=1)  # 预测的图像通常是一个多类概率分布（例如 softmax 输出）。torch.argmax 会沿着类别维度（dim=1）选择具有最大概率的类别，这样得到每个像素的预测标签。
		pred_graph = pred_graph.cpu().numpy()  # 将预测标签从 GPU 转移到 CPU，并转换为 NumPy 数组。
		gt_graph_list.append(gt_graph)
		pred_graph_list.append(pred_graph)


	gt_graph_list = List2List(gt_graph_list)
	pred_graph_list = List2List(pred_graph_list)

	# 打印出 gt_graph_list 和 pred_graph_list 中唯一的值
	print("Unique values in gt_graph_list:", np.unique(gt_graph_list))
	print("Unique values in pred_graph_list:", np.unique(pred_graph_list))

	# Overall F1 for graph  在微平均下计算 F1 分数。这意味着在计算 F1 分数时，首先对所有类别的真实值和预测值进行累加，然后基于整体的 TP（True Positive）、FP（False Positive）、FN（False Negative）等指标计算 F1 分数。
	f1_graph_overall = f1_score(gt_graph_list, pred_graph_list, average='micro')
	# Mean F1 for graph  在宏平均下计算 F1 分数。这意味着先计算每个类别的 F1 分数，然后取平均。这个方法不会考虑类别不平衡的影响。
	f1_graph_mean = f1_score(gt_graph_list, pred_graph_list, average='macro')

	return f1_graph_overall, classification_report(gt_graph_list, pred_graph_list), f1_graph_mean


def Cal_IoU(total_gt_bev, total_pred_bev):
	# confmat = ConfusionMatrix(3)
	confmat = ConfusionMatrix(5)
	for idx in range(len(total_gt_bev)):
		confmat.update(total_gt_bev[idx].flatten(), total_pred_bev[idx].argmax(1).flatten())
	confmat.reduce_from_all_processes()

	return confmat

class ConfusionMatrix(object):
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.mat = None

	def update(self, a, b):
		n = self.num_classes
		if self.mat is None:
			self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
		with torch.no_grad():
			k = (a >= 0) & (a < n)
			inds = n * a[k].to(torch.int64) + b[k]
			self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

	def reset(self):
		if self.mat is not None:
			self.mat.zero_()

	def compute(self):
		h = self.mat.float()
		acc_global = torch.diag(h).sum() / h.sum()
		acc = torch.diag(h) / h.sum(1)
		iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
		return acc_global, acc, iu

	def reduce_from_all_processes(self):
		if not torch.distributed.is_available():
			return
		if not torch.distributed.is_initialized():
			return
		torch.distributed.barrier()
		torch.distributed.all_reduce(self.mat)

	def __str__(self):
		acc_global, acc, iu = self.compute()
		return (
			'global correct: {:.1f}\n'
			'average row correct: {}\n'
			'IoU: {}\n'
			'mean IoU: {:.1f}').format(
				acc_global.item() * 100,
				['{:.1f}'.format(i) for i in (acc * 100).tolist()],
				['{:.1f}'.format(i) for i in (iu * 100).tolist()],
				iu.mean().item() * 100)
