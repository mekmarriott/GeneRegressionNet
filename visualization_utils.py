import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_performance(labels, survival, predictions, title='Predictions vs. Ground Truth'):
	dead_indices = [i for i in range(len(labels)) if survival[i] == 1]
	alive_indices = [i for i in range(len(labels)) if survival[i] == 0]
	max_num = int(max(np.max(labels), np.max(predictions))) + 1
	line = np.array(range(max_num))
	plt.plot(line, line)
	# TODO: Make different colors
	plt.scatter(np.array(labels)[alive_indices], np.array(predictions[alive_indices]), marker="o")
	plt.scatter(np.array(labels)[dead_indices], np.array(predictions[dead_indices]), marker=">")
	plt.xlim([0,max_num])
	plt.ylim([0,max_num])
	plt.xlabel('Ground Truth')
	plt.ylabel('Predictions')
	plt.title(title)
	plt.show()

def plot_train_test_performance(train_feed, test_feed, train_pred, test_pred, title='Predictions vs. Ground Truth', save_fig=None):
	f, (train_plt, test_plt) = plt.subplots(1, 2, sharey=True)
	# Plot train performance
	dead_indices = [i for i in range(len(train_feed['y'])) if train_feed['d'][i] == 1]
	alive_indices = [i for i in range(len(train_feed['y'])) if train_feed['d'][i] == 0]
	train_plt.scatter(train_feed['y'][dead_indices], train_pred[dead_indices], c='r', marker="x", label="Dead")
	train_plt.scatter(train_feed['y'][alive_indices], train_pred[alive_indices], c='b', label="Alive")
	train_plt.set_title('Training Calibration')
	# Plot test performance
	dead_indices = [i for i in range(len(test_feed['y'])) if test_feed['d'][i] == 1]
	alive_indices = [i for i in range(len(test_feed['y'])) if test_feed['d'][i] == 0]
	test_plt.scatter(train_feed['y'][dead_indices], test_pred[dead_indices], c='r', marker="x", label="Dead")
	test_plt.scatter(train_feed['y'][alive_indices], test_pred[alive_indices], c='b', label="Alive")
	test_plt.set_title('Test Calibration')
	# Plot other stuff
	max_train = max(np.max(train_feed['y']), np.max(train_pred))
	max_test = max(np.max(test_feed['y']), np.max(test_pred))
	max_num = int(max(max_train, max_test)) + 1
	plt.xlim([0,max_num])
	plt.ylim([0,max_num])
	line = np.array(range(max_num))
	plt.xlabel("Ground Truth")
	plt.ylabel("Prediction")
	plt.legend(loc=1)
	plt.legend(loc=2)
	train_plt.plot(line, line)
	test_plt.plot(line, line)

	plt.show()

def plot_by_num_mutations(indices):
	mutation_count = defaultdict(int)
	for idx in indices.flatten():
		mutation_count[idx] += 1
	print mutation_count