import numpy as np
from scipy.stats import entropy
from clusterer import Clusterer
import pickle as pkl
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
from collections import Counter
from networkx.algorithms import approximation
import networkx.algorithms.community as nx_comm
import time 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_misclassification(data, save_dir):
    prop_vals1 = []
    prop_vals2 = []
    methods = []
    num_ht = len(np.where(data.y.numpy()==1)[0])
    num_isw = len(np.where(data.y.numpy()==2)[0])

    cm = confusion_matrix(data.y.numpy(), data.prob_labels.argmax(1))
    ht_acc = cm[1][1]/num_ht
    num_isw_ht = cm[2][1]
    isw_mis = num_isw_ht/num_isw

    prop_vals1.append(isw_mis)
    prop_vals2.append(ht_acc)
    methods.append("MV")

    for method in ['mlp', 'gcn', "nrgnn", 'pignn', 'tnet_cl']:
        # model_save_name = save_dir+"/"+method+"_results_best_model.pkl"
        # best_model = torch.load(model_save_name)
        # run the model on the entire dataset
        # if method == 'nrgnn':
        #     results, cm = best_model.test(data.y, np.array(range(len(data.x))))
        # elif method == 'pignn':
        #     _, pred = best_model(data)[0].max(dim=1)
        #     cm = confusion_matrix(data.y, pred)
        # elif method == 'mlp':
        #     pred = best_model.predict(data.x)
        #     cm = confusion_matrix(data.y, pred)
        # else:
        #     results = best_model.test_model(data, np.array(range(len(data.x))))
        #     cm = confusion_matrix(data.y, results['preds'])

        if method == 'nrgnn':
            model_save_name = "baselines/NRGNN/nrgnn_best_model.pkl"
            best_model = torch.load(model_save_name)
            res, cm = best_model.test(data.y, np.array(range(len(data.x))))
            # cm = confusion_matrix(data.y, results['preds'])
        elif method == 'mlp':
            cm = pkl.load(open("results/synthetic_asw/mlp_results_confusion_matrix.pkl",'rb'))
        elif method == 'pignn':
            cm = pkl.load(open("baselines/pi_gnn/pigcn_confusion_matrix.pkl",'rb'))
            method = 'pignn'
        elif method == 'gcn':
            model_save_name = "results/synthetic_asw/"+method+"_results_best_model.pkl"
            best_model = torch.load(model_save_name)
            results = best_model.test_model(data, np.array(range(len(data.x))))
            cm = confusion_matrix(data.y, results['preds'])
        else:
            model_save_name = "results/synthetic_asw/"+method+"_results_best_model.pkl"
            if method in ['tnet_cl_no_struct','tnet_cl']:
                method = 't-net'
                j=2

            best_model = torch.load(model_save_name)
            results = best_model.test_model(data, np.array(range(len(data.x))))
            cm = confusion_matrix(data.y, results['preds'])

        ht_acc = cm[1][1]/num_ht
        num_isw_ht = cm[2][1]
        isw_mis = num_isw_ht/num_isw
        prop = isw_mis

        prop_vals1.append(prop)
        prop_vals2.append(ht_acc)
        methods.append(method.upper())

    
    width=0.35/2
    plt.bar(x=[x-width/2 for x in range(len(prop_vals1))], height=prop_vals1, width=width, color='red', alpha=0.8, label='ISW conflict')
    plt.bar(x=[x+width/2 for x in range(len(prop_vals2))], height=prop_vals2, width=width, alpha=0.8, label='HT accuracy')
    plt.xticks(ticks=range(len(prop_vals1)), labels=methods)
    plt.legend(loc=(0.50, 0.75))
    # plt.title("Misclassified ISW and HT Accuracy")
    plt.show()

def print_tabular_results(save_dir):
    # function to print the results of all methods for table
    import os
    print("Displaying classification results....\n")
    for file in os.listdir(save_dir):
        if file.endswith("_results.pkl"):
            full_path = os.path.join(save_dir, file)
            results = pkl.load(open(full_path, 'rb'))
            print(file.split("results")[0]+" : ")

            for label in ['HT', 'ISW', 'Spam', 'weighted avg']:
                for metric in ['precision', 'recall', 'f1-score']:
                    res = []
                    for item in results: # iterating through list of 5 seed runs
                        res.append(item[label][metric])
                    print(label + " " + metric + " " + str(np.mean(res)) + " " + str(np.std(res)))
            print("\n")

def print_pretty_results(results):
    clean_res1 = []
    prec = []
    rec = []
    for item in results:
        clean_res1.append(item['weighted avg']['f1-score'])
        prec.append(item['weighted avg']['precision'])
        rec.append(item['weighted avg']['recall'])

    print("\nFinal:")
    print(np.mean(clean_res1), np.std(clean_res1))

def get_probabilistic_labels(weak_labels, num_classes):
    # converts the weak labels to probabilistic labels
    prob_labels = []
    num_lfs = len(weak_labels[0])
    weak_labels = np.array(weak_labels)
    for weak_label in weak_labels[:,:-1]:
        p_labels = [0]*num_classes
        frequency_of_lf_votes = Counter(weak_label).most_common()
        if frequency_of_lf_votes[0][0] != -1:
            mv_label = frequency_of_lf_votes[0][0]
        elif len(frequency_of_lf_votes)==1: # all LFs returned -1
            mv_label = np.random.choice(range(num_classes))
        else:
            mv_label = frequency_of_lf_votes[1][0]

        p_labels[int(mv_label)] = 1

        prob_labels.append(p_labels)
    return np.array(prob_labels)


def load_data(filepath, epochs):
    # loads the data file from the given filepath
    data = pkl.load(open(filepath,'rb'))
    num_classes = len(np.unique(data.y))
    data.num_classes = num_classes
    data.num_features = len(data.x[0])
    num_times = epochs
    try:
        if len(data.prob_labels) == len(data.x):
            print("Loading previously saved data..")
            return data
    except:
        print("Calculating probabilistic labels and communities...")
        data.prob_labels = get_probabilistic_labels(data.weak_labels, num_classes)

    data.edge_index = torch.LongTensor(data.edge_index)
    data.x = torch.FloatTensor(data.x)
    data.y = torch.LongTensor(data.y)
    nx_graph = to_networkx(data, to_undirected=False)
    data.nx_graph = nx_graph
    data.adj = nx.adjacency_matrix(nx_graph)
    print("Computing communities....")
    # data.community_mapping = louvain_communities.best_partition(nx_graph)
    data.communities = nx_comm.louvain_communities(nx_graph)

    print("Num communites = ", str(len(data.communities)))
    print("Num nodes = ", str(len(data.x)))
    print("Num feats = ", str(data.num_features))

    comm_map = {}
    comm_sizes = []
    degree_of_nodes_within_comm = {}
    for ind, cc in enumerate(data.communities):
        comm_sizes.append(len(cc))
        degree_of_nodes_within_comm[ind] = [x[1] for x in list(nx_graph.subgraph(cc).degree)]
        for node in cc:
            comm_map[node] = ind
    data.community_mapping = comm_map
    data.degree_of_nodes_within_comm = degree_of_nodes_within_comm

    comm_nbrs = {}
    community_pos_options = []

    for node in tqdm(data.nx_graph.nodes):
        comm = data.community_mapping[node]
        comm_nodes = data.communities[comm].copy()
        if len(comm_nodes) != 1:
            comm_nodes.remove(node)
        comm_nbrs[node] = comm_nodes
        community_pos_options.append(np.random.choice(list(comm_nodes), size=num_times))
    
    data.community_pos_options = np.array(community_pos_options)

    pkl.dump(data, open(filepath,'wb'))
    return data

def calculate_final_weights(data, embeddings):
    # calculates the final weight given entropy and node importance values
    inf_s = structural_node_importance(data.community_mapping, data.communities, data.nx_graph, data.degree_of_nodes_within_comm)
    inf_e = embedding_node_importance(embeddings, data.num_classes)
    E = calculate_lf_agreement(data.weak_labels, data.num_classes)
    scores = []
    for ent, struct_imp, emb_imp in zip(E, inf_s, inf_e):
        # scores.append(ent * (struct_imp + emb_imp))
        scores.append(ent*emb_imp)
    return torch.tensor(scores)

def structural_node_importance(comm_mapping, graph_comms, nx_graph, degree_of_nodes_within_comm):
    # calculates the strucutral importance of each node in the graph
    '''
    params:
            graph_comms - set of nodes in each community
            comm_mapping - Louvain communities present in the graph. {node: comm} format
            nx_graph - networkx graph object
    returns:
            inf_s - influence/importance of node based on its position in the graph
    '''
    inf_s = []
    for node in nx_graph.nodes:
    # for node, comm in tqdm(comm_mapping.items()):
        comm = comm_mapping[node]
        community_size = len(graph_comms[comm])
        degree_in_community = nx_graph.subgraph(graph_comms[comm]).degree[node]
        sum_degree_of_nodes_within_comm = sum(degree_of_nodes_within_comm[comm])
        degree_in_graph = nx_graph.degree(node)
        # score = community_size * (degree_in_community / (degree_in_graph - degree_in_community + 10e-5))
        if sum_degree_of_nodes_within_comm == 0:
            score = 0
        else:
            score = community_size * degree_in_community / sum_degree_of_nodes_within_comm
        inf_s.append(score)

    # inf_s = [ee/np.mean(inf_s) for ee in inf_s]
    return np.array(inf_s)

def embedding_node_importance(node_embeddings, num_classes):
    # calculates the node importance based on distance of embedding from kmeans cluster centroid
    '''
    params:
            node_embeddings - node embeddings that are robust to weak labels (learned using graph contrastive learning)
    returns:
            inf_e - node influence score based on its embeddings
    '''
    clustering_module = Clusterer(name='Kmeans')
    clustering_module.cluster_kmeans(node_embeddings.detach().numpy(), num_clusters=num_classes)
    dist_from_centroids = clustering_module._return_dists_from_centroid(node_embeddings.detach().numpy())
    cluster_labels = clustering_module._return_labels()
    inf_e = []
    sum_dist_from_centroids = sum(dist_from_centroids)

    for node_id in range(len(dist_from_centroids)):
        node_cluster = cluster_labels[node_id]
        cluster_size = len(np.where(cluster_labels == node_cluster)[0])
        dist_from_centroid = dist_from_centroids[node_id]
        inf_e.append(cluster_size * dist_from_centroid / sum_dist_from_centroids)
        # inf_e.append(cluster_size/dist_from_centroid)

    # inf_e = normalize(np.array(inf_e).reshape(1,-1), norm='l1', axis=0)
    # inf_e = [ee/np.mean(inf_e) for ee in inf_e]
    return np.array(inf_e)

def calculate_lf_agreement(lfs, num_classes):
    # calculates the entropy of the LF labels as a measure of their agreement/disagreement
    '''
    params:
            lfs - nxm matrix which shows the output of m LFs for n nodes in the graph
    returns:
            E - entropy of LFs
    '''
    lfs = np.array(lfs,dtype=str)
    counts = np.array([np.unique(lf_row, return_counts=True)[1] for lf_row in lfs])
    ents = [entropy(count) for count in counts]
    upper_bound = np.log(num_classes)
    E = [upper_bound-e for e in ents]
    # E = [1/(e*e + 10e-5) for e in ents]
    # E = [ee/np.mean(E) for ee in E]
    return np.array(E)