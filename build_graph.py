'''
This file contains the code to convert the csv files into graph data for node classification using GCN.
Usage: python build_graph.py
Author: Pratheeksha Nair
'''
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from itertools import combinations
import torch_geometric.data as make_dataset
from labeling_functions import apply_lfs


def find_meta_clusters(df):
	# function to find meta-cluster labels
	print("Finding meta clusters...\n")
	num_micro = df['LSH label'].nunique()
	clus_ind_map = dict(zip(df['LSH label'].unique(),range(num_micro)))
	micro_to_meta_map = np.zeros([num_micro, num_micro])
	p = df.dropna(subset=['phone_num_cleaned_str'])
	for id, grp in tqdm(p.groupby('phone_num_cleaned_str')):
	# p = df.dropna(subset=['phone'])
	# for id, grp in tqdm(p.groupby('phone')):
	    clusters = grp['LSH label'].unique()
	    pairs = combinations(clusters, 2)
	    for e1, e2 in pairs:
	        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1
	
	p = df.dropna(subset=['image_id'])  
	for id, grp in tqdm(p.groupby('image_id')):
	    clusters = grp['LSH label'].unique()
	    pairs = combinations(clusters, 2)
	    for e1, e2 in pairs:
	        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	p = df.dropna(subset=['email'])
	for id, grp in tqdm(p.groupby('email')):
	    clusters = grp['LSH label'].unique()
	    pairs = combinations(clusters, 2)
	    for e1, e2 in pairs:
	        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	p = df.dropna(subset=['social'])
	for id, grp in tqdm(p.groupby('social')):
	    clusters = grp['LSH label'].unique()
	    pairs = combinations(clusters, 2)
	    for e1, e2 in pairs:
	        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	nx_graph = nx.from_numpy_matrix(micro_to_meta_map).to_directed()
	# finding connected components as meta clusters
	meta_label_map = {}
	clus_counter = 0
	num_comps = 0
	for compo in nx.strongly_connected_components(nx_graph):
	    num_comps += 1
	    for node in list(compo):
	        meta_label_map[node] = clus_counter
	    clus_counter += 1
	df['Meta label'] = df['LSH label'].apply(lambda x:meta_label_map[clus_ind_map[x]])
	
	return df, nx_graph

def preprocess(df, cities):
	# cities = cities[cities.country_id==3]
	df = pd.merge(df, cities, left_on='city_id', right_on='id', how='left')
	df.rename(columns={'phone':'phone_num', 'body':'description'}, inplace=True)
	return df

def get_weak_labels(data):
	# weak_labels = pickle.load(open("marinus_labelled/merged_data_3_class_no_dupl_names_LSH_labels_weak_labels2.pkl",'rb'))
	if 'geolocation' not in data.columns:
		cities = pd.read_csv("marinus_labelled/cities.csv", index_col=False)
		data = preprocess(data, cities)
	label_mat, spa_clusters, lf_votes = apply_lfs(data, level_of_analysis='LSH label')

	clus_ind_map = dict(zip(data['LSH label'].unique(),range(data['LSH label'].nunique())))
	lambda_mat = np.zeros(shape=[data['LSH label'].nunique(), 12])
	for lsh_label, per_class_labels in lf_votes.items():
		# 0 - SPAM
		# 1 - HT
		# 2 - ISW
		row_indx = clus_ind_map[lsh_label]
		col_indx = 0
		for vote in per_class_labels['ht']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 1
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[1,0],size=1)[0]
				chosen_label = -1
				lambda_mat[row_indx, col_indx] = chosen_label # abstaining or ISW
			col_indx += 1
		for vote in per_class_labels['isw']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 2
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[0.8,0.2],size=1)[0]
				lambda_mat[row_indx, col_indx] = -1 # abstaining or ISW
			col_indx += 1
		for vote in per_class_labels['spam']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 0
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[1,0],size=1)[0]
				chosen_label = -1
				lambda_mat[row_indx, col_indx] = chosen_label # abstaining or ISW
			col_indx += 1

	return lambda_mat

def get_labels(df):
	clus_ind_map = dict(zip(df['LSH label'].unique(),range(df['LSH label'].nunique())))
	label_df = df[['LSH label', 'label']].drop_duplicates().to_numpy()
	label_dict = dict(label_df)
	labels = np.zeros(len(label_dict))
	for lsh_label, class_label in label_dict.items():
		labels[clus_ind_map[lsh_label]] = class_label

	return labels

def get_graph(nx_graph, y, feats, df):
	edge_index = [[],[]]
	for line in nx.generate_edgelist(nx_graph, data=False):
		edge_index[0].append(int(line.split()[0]))
		edge_index[1].append(int(line.split()[1]))

	data = make_dataset.Data(x=feats, y=y, edge_index=edge_index)
	data.weak_labels = get_weak_labels(df)
	return data

def modify_feats(feat_df, clus_ind_map):
	cols_to_keep = []
	for col in feat_df.columns:
		if 'Val' not in col:
			cols_to_keep.append(col)

	feats = feat_df[cols_to_keep].to_numpy()
	modified_feats = np.zeros(shape=[len(feats), len(feats[0])-1])
	for row in feats:
		if row[-1] == -1:
			continue
		indx = clus_ind_map[row[-1]]
		modified_feats[indx] = row[:-1]

	return modified_feats


def get_data_df():

	# data = pd.read_csv("marinus_labelled/merged_data_3_class_no_dupl_names_LSH_labels.csv", index_col=False)
	data = pd.read_csv("marinus_canada/HT2018_final_trimmed_for_labeling_neat_preprocessed.csv", index_col=False)
	
	data, nx_graph = find_meta_clusters(data)
	feat_df = pd.read_csv("marinus_canada/plot_df.csv", index_col=False)
	clus_ind_map = dict(zip(data['LSH label'].unique(),range(data['LSH label'].nunique())))
	feats = modify_feats(feat_df, clus_ind_map)

	# labels = get_labels(data)
	labels = []
	data_graph = get_graph(nx_graph, labels, feats, data)
	print(data['LSH label'].nunique(), feat_df.shape)

	# pickle.dump(data_graph, open("marinus_labelled/marinus_labelled_graph.pkl",'wb'))


if __name__ == '__main__':
	get_data_df()

