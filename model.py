import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv, GCNConv, SimpleConv
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report, fbeta_score
from utils import calculate_final_weights
from contrastive_loss import ContrastiveLoss

class GCN(torch.nn.Module):
    def __init__(self, K, input_dim, output_dim, conv_model='gcn'):
        super(GCN, self).__init__()
        self.conv_model = conv_model
        self.input_dim = input_dim
        self.embedding_dim = 64
        self.output_dim = output_dim
 
        if conv_model == 'sgc':
            self.conv1 = SGConv(
                        input_dim, output_dim, K=K, cached=True)
        elif conv_model == 'gcn':
            self.conv1 = GCNConv(input_dim, self.embedding_dim, \
                improved=True, cached=False, add_self_loops=True, normalize=True)
            self.dropout = torch.nn.Dropout(p=0.5)
            self.conv2 = GCNConv(self.embedding_dim, output_dim, \
                improved=True, cached=False, add_self_loops=True, normalize=True)

    def set_attributes(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0005)
        self.optimizer = optimizer
    
    @staticmethod
    def create_model(input_dimension, output_dimension, conv_model='gcn'):
        # function to build the required model
        model = GCN(K=16, input_dim=input_dimension, output_dim=output_dimension, conv_model=conv_model)      
        model.set_attributes()
        return model

    def get_embedding(self):
        return self.embedding

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        if self.conv_model == 'gcn':
            x = F.relu(x)
            self.embedding = x
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

    def train_model(self, data, train_mask, iter_n):
        self.train()
        self.optimizer.zero_grad()

        preds = self.forward(data)
        pred_labels = preds.argmax(1)

        loss = F.nll_loss(preds[train_mask], data.y[train_mask])
        loss.backward()

        self.optimizer.step()
        return loss.detach().item(), pred_labels

    def test_model(self, data, test_mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            f1_scores = []
            pred_labels = logits[test_mask].max(1)[1].cpu().numpy()
            labels = data.y[test_mask].cpu().numpy()
            f1_scores = {
                    'preds': pred_labels,
                    'weighted': f1_score(labels, pred_labels, average='weighted'),
                    'fbeta': fbeta_score(labels, pred_labels, average='weighted',beta=0.5,labels=[1])
                }

        return f1_scores['class_report'], confusion_matrix(labels, pred_labels)

class GCNNet(torch.nn.Module):
    def __init__(self, K, input_dim, output_dim, conv_model='gcn'):
        super(GCNNet, self).__init__()
        self.conv_model = conv_model
        self.input_dim = input_dim
        self.embedding_dim = 64
        self.output_dim = output_dim
  
        if conv_model == 'sgc':
            self.conv1 = SGConv(
                        input_dim, output_dim, K=K, cached=True)
        elif conv_model == 'gcn':
            # hidden size of 64 as in the GPA paper 
            self.aggregate_layer = SimpleConv(aggr='mean', requires_grad=True)
            self.conv1 = GCNConv(input_dim, self.embedding_dim, \
                improved=True, cached=False, add_self_loops=True, normalize=True)
            self.dropout = torch.nn.Dropout(p=0.5)
            self.conv2 = GCNConv(self.embedding_dim, output_dim, \
                improved=True, cached=False, add_self_loops=True, normalize=True)

    def set_attributes(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0005)
        self.optimizer = optimizer
    
    @staticmethod
    def create_model(input_dimension, output_dimension, conv_model='gcn'):
        # function to build the required model
        model = GCNNet(K=16, input_dim=input_dimension, output_dim=output_dimension, conv_model=conv_model)      
        model.set_attributes()
        return model

    def get_negative_embeddings(self, data):
        neg_node_feats = data.x[torch.randperm(data.x.size()[0])]
        aggregated_embedding = self.aggregate_layer(neg_node_feats, data.edge_index)
        self.negative_embedding = self.conv1(aggregated_embedding, data.edge_index)

    def get_embedding(self):
        return self.embedding

    def forward(self, data, init_embedding):
        if init_embedding is None:
            x = data.x
        else:
            x = init_embedding
        edge_index = data.edge_index

        # if self.conv_model=='gcn':
        #     x = self.aggregate_layer(x, edge_index)

        if init_embedding is None:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        if self.conv_model == 'gcn':
            self.embedding = x
            x = self.conv2(x, edge_index)
            # x = F.relu(x)
        
        return F.log_softmax(x, dim=1)


    def train_model(self, data, train_mask, iter_n, init_embedding=None):
        self.train()
        self.optimizer.zero_grad()

        preds = self.forward(data, init_embedding)
        pred_labels = preds.argmax(1)

        weak_labels = torch.tensor(data.prob_labels).argmax(1)

        loss2 = F.nll_loss(preds[train_mask], weak_labels[train_mask], reduction='none')
        node_weights = calculate_final_weights(data, self.get_embedding())[train_mask]

        loss2 = loss2 * node_weights
        loss2 = loss2.mean()

        # if self.conv_model=='gcn':
        #     self.get_negative_embeddings(data)
        #     loss1 = ContrastiveLoss()(self.embedding, data.community_pos_options, \
        #         self.negative_embedding, iter_n)
        #     loss = loss1+loss2
        # else:
        loss = loss2
        loss.backward()

        self.optimizer.step()
        return loss.detach().item(), pred_labels


    def test_model(self, data, test_mask, init_embedding=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(data, init_embedding)
            f1_scores = []
            pred_labels = logits[test_mask].max(1)[1].cpu().numpy()
            labels = data.y[test_mask].cpu().numpy()
            f1_scores = {
                    'preds': pred_labels,
                    'weighted': f1_score(labels, pred_labels, average='weighted'),
                    'class_report': classification_report(labels, pred_labels, labels=[1,2,0], target_names=['HT',"ISW",'Spam'], output_dict=True)
                }
        return f1_scores
