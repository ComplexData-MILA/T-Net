import argparse
import pandas as pd
import numpy as np
from utils import *
from model import GCNNet, GCN
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from run_baselines import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rcParams.update({'font.size': 28, 'font.family':'Times New Roman'})


def save_results(results, save_dir, filename):
	pkl.dump(results, open(save_dir+"/"+filename,'wb'))


def parse_args():
	parser = argparse.ArgumentParser(description='MO Detection')
	parser.add_argument('--data_file', type=str, 
					help='path to the pickle file containing the graph')
	parser.add_argument("--epochs", type=int, default=100,
					help='number of training epochs')
	parser.add_argument("--save_dir", type=str,
					help='path to directory for saving results')
	parser.add_argument("--save_filename", type=str,
					help='name of file to save results')
	parser.add_argument("--baseline", action='store_true',
					help="indicate whether or not to run a baseline model instead of T-Net")
	parser.add_argument("--baseline_method", choices=['mlp','mv', 'pigcn', 'nrgnn', 'dgnn'],
					help="indicate which baseline to run. Only works if --baseline is True")
	parser.add_argument("--print_results", action='store_true', default=False, 
					help='print results for table. Can only be set to true after all baselines have been run')
	parser.add_argument("--saved_model_path", type=str, default='',
					help='path to the saved model file')
	parser.add_argument("--get_misclassification", action='store_true', default=False,
					help='Indicate whether to plot the ISW misclassification rate')
	args = parser.parse_args()
	return args


def train_classifier(data, save_dir, save_filename, epochs=100, gcn_pretrain=False):
	# this function is used to train the classifier
	kf = KFold(n_splits=5)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	num_features = len(data.x[0])
	num_classes = len(np.unique(data.y))
	all_results = []
	model_save_name = save_filename.split(".pkl")[0]+"_best_model.pkl"

	for train_mask, test_mask in kf.split(list(range(len(data.x)))):
		if gcn_pretrain:
			model = GCN.create_model(num_features, num_classes,conv_model='gcn').to(device)
		else:
			model = GCNNet.create_model(num_features, num_classes).to(device)
		losses = []
		best_test_f1 = 0.
		best_model = None
		for epoch in range(1, epochs + 1):
			loss, predicted_labels = model.train_model(data, train_mask, epoch-1)
			train_f1 = f1_score(data.y[train_mask], predicted_labels[train_mask], average='weighted')

			test_f1 = model.test_model(data, test_mask)['class_report']
			test_f1 = test_f1['weighted avg']['f1-score']
			print("Epoch : {:02d}, Loss : {:.4f}, Train F1 score : {:.4f}, Test F1 score : {:.4f}".format(epoch, loss, train_f1, \
				test_f1)) 
			losses.append(loss)
			if test_f1 > best_test_f1:
				best_test_f1 = test_f1
				torch.save(model, save_dir+"/"+model_save_name)

		best_model = torch.load(save_dir+"/"+model_save_name)
		final_test_f1 = best_model.test_model(data, test_mask)['class_report']
		print("Test f1: ", final_test_f1['weighted avg']['f1-score'])
		all_results.append(final_test_f1)

	save_results(all_results, save_dir, save_filename)
	return all_results

def main():
	args = parse_args()
	data = load_data(args.data_file, args.epochs) # loading the data file

	if args.get_misclassification: # plot the ISW conflict rate and HT accuracy for all methods
		plot_misclassification(data, args.save_dir)
		return 

	if args.print_results: # display results
		print_tabular_results(args.save_dir)
		return

	if args.baseline: # run baseline
		if args.baseline_method == 'mlp':
			classification_results = train_mlp_classifier(data, args.save_dir, args.save_filename)
		elif args.baseline_method == 'mv': # run MV basline
			classification_results = mv_baseline(data, args.save_dir, args.save_filename)
		elif args.baseline_method == 'gcn': # run GCN baseline
			classification_results = train_classifier(data, args.save_dir, args.save_filename, \
				epochs=args.epochs, gcn_pretrain=True)
		elif args.baseline_method == 'pigcn': # run PI-GCN baseline
			classification_results = pigcn_baseline(data, args)
		elif args.baseline_method == 'nrgnn': # run NRGNN baseline
			classification_results = nrgnn_baseline(data, args.save_dir, args.save_filename)
		elif args.baseline_method == 'dgnn': # run NRGNN baseline
			classification_results = dgnn_baseline(data, args.save_dir, args.save_filename)
	else:
		classification_results = train_classifier(data, args.save_dir, args.save_filename, epochs=args.epochs)

	print_pretty_results(classification_results)


if __name__ == '__main__':
	main()