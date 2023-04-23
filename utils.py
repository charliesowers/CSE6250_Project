"""
Most of the codes here are from HW 5
"""

import time
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, precision_recall_curve
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():
		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size

def compute_batch_rocAuc(output, target):
	"""
	compute the roc auc for a batch
	"""

	with torch.no_grad():
		_, pred = output.max(1)
		auc = roc_auc_score(target, pred)
		return auc


def train_RNN(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	auc = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)

		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
		auc.update(compute_batch_rocAuc(output, target).item(), target.size(0))
		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, auc.avg

def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	auc = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
			auc.update(compute_batch_rocAuc(output, target).item(), target.size(0))
			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, auc.avg, results

# from homework 5
class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        self.seqs = []
        for seq in seqs:
            num_visits = len(seq)
            row = []
            col = []
            for i in range(num_visits):
                row.extend([i]*len(seq[i]))
                col.extend(seq[i])
            data = [1] * len(col)
            sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=(num_visits, num_features))
            self.seqs.append(sparse_matrix)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """

    # TODO: Return the following two things
    # TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
    # TODO: 2. Tensor contains the label of each sequence
    tuple_list = [(i, t.shape[0]) for i, (t, _) in enumerate(batch)]
    sorted_tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

    max_row = sorted_tuple_list[0][1]
    max_col = batch[0][0].shape[1]

    seqs, lens, labels = [], [], []
    for i, y in sorted_tuple_list:
        lens.append(y)
        labels.append(batch[i][1])
        old_matrix = batch[i][0].toarray()
        row_num = max_row - y
        new_matrix = np.zeros((row_num, max_col))
        final_matrix = np.concatenate([old_matrix, new_matrix])
        seqs.append(final_matrix)

    seqs_tensor = torch.FloatTensor(seqs)
    lengths_tensor = torch.LongTensor(lens)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor
