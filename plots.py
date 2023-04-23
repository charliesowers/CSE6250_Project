import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, path, name):
	# plot loss curves
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Training loss')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss curve')
	plt.legend()
	plt.savefig(bbox_inches='tight', fname=f'{path}/{name}_loss.png')

	# plot accuracy curve
	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training accuracy')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy curve')
	plt.legend()
	plt.savefig(bbox_inches='tight', fname=f'{path}/{name}_accuracy.png')

def plot_confusion_matrix(results, class_names, path, name):
	y_true, y_pred = zip(*results)
	m = confusion_matrix(y_true, y_pred, normalize='true')
	disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=class_names)
	disp.plot(xticks_rotation='vertical')
	plt.savefig(bbox_inches='tight', fname=f'{path}/{name}.png')
