from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle as pkl
from baselines.pi_gnn.model.PI_GCN import run_pi_gcn
from baselines.NRGNN.train_NRGNN import run_nrgnn
from baselines.denoising_gnn.run_denoising_gnn import run_dgnn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report, confusion_matrix, fbeta_score
from collections import Counter

def save_results(results, save_dir, filename):
	pkl.dump(results, open(save_dir+"/"+filename,'wb'))

def dgnn_baseline(data, save_dir, save_filename):
	results = run_dgnn(data)
	save_results(results, save_dir, save_filename)
	return results

def nrgnn_baseline(data, save_dir, save_filename):
	results = run_nrgnn(data, save_dir, save_filename)
	save_results(results, save_dir, save_filename)
	return results

def pigcn_baseline(data, args):
	results = run_pi_gcn(data, args.save_dir, args.save_filename, epochs=args.epochs)
	save_results(results, args.save_dir, args.save_filename)
	return results

def mv_baseline(data, save_dir, save_filename):
	kf = KFold(n_splits=5)
	all_results = []
	for train_mask, test_mask in kf.split(list(range(len(data.x)))):
		y_prob = data.prob_labels.argmax(1)[test_mask]
		all_results.append(classification_report(data.y[test_mask], y_prob, labels=[1,2,0], target_names=['HT', 'ISW', 'Spam'], \
			output_dict=True))
		all_results['fbeta'] = fbeta_score(data.y[test_mask], y_prob, beta=0.5, average='weighted')
	save_results(all_results, save_dir, save_filename)
	return all_results

def train_mlp_classifier(data, save_dir, save_filename):
	kf = KFold(n_splits=5)
	all_results = []
	for train_mask, test_mask in kf.split(list(range(len(data.x)))):
		clf = MLPClassifier(random_state=1, max_iter=800).fit(data.x[train_mask], data.prob_labels.argmax(1)[train_mask])
		y_pred= clf.predict(data.x[test_mask])
		y_prob = data.prob_labels.argmax(1)[test_mask]
		results = classification_report(data.y[test_mask].numpy(), y_pred, labels=[1,2,0], target_names=['HT',"ISW",'Spam'], \
			output_dict=True)
		results['fbeta'] = fbeta_score(data.y[test_mask].numpy(), y_pred, average='weighted',beta=0.5, labels=[0])
		all_results.append(results)

	pkl.dump(clf, open(save_dir+"/"+save_filename.split('.pkl')[0]+"_best_model.pkl",'wb'))	
	save_results(all_results, save_dir, save_filename)
	return all_results