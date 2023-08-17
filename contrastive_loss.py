import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from torch.autograd import Variable

tau = 0.65
num_negative_samples_network = 10


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.agg_mi_score = torch.vmap(self.compute_mutual_info_score)

    def compute_mutual_info_score(self, embedding, positive_example, all_samples, weight_matrix=None):
        if weight_matrix != None:
            transformed_embedding = torch.matmul(embedding,weight_matrix).detach()
            numerator = torch.matmul(transformed_embedding, positive_example.t())/tau

            denominator = torch.stack([torch.matmul(transformed_embedding,selected_feat)/tau \
                for selected_feat in all_samples],dim=0)
            denominator_sum = torch.logsumexp(denominator, dim=0)
        else:
            numerator = torch.matmul(embedding.detach(), positive_example.t())/tau
            denominator = torch.stack([torch.matmul(embedding.detach(), selected_feat.t())/tau \
                for selected_feat in all_samples], dim=0)
            denominator_sum = torch.logsumexp(denominator, dim=0)

        score = denominator_sum - numerator
        return score 

    def network_stucture_loss(self, embeddings, negative_embs, community_pos_options, iter_n):
        loss_sums = []
        num_samples = len(embeddings)
        nodes = np.array(list(range(num_samples)))

        positive_example_ind = community_pos_options[nodes][:,iter_n-1]
        positive_examples = torch.unsqueeze(embeddings[positive_example_ind],-1).permute(0,-1, 1)

        all_negatives = []
        for sample in range(num_negative_samples_network):
            negative_example_ind = np.random.choice(list(range(len(negative_embs))), size=num_samples) # repetitions possible, no way around it!
        
            neg_examples = negative_embs[negative_example_ind].t()
            all_negatives.append(neg_examples)
        all_negatives = torch.stack(all_negatives).permute(2,0,1)
        all_samples = torch.hstack([positive_examples, all_negatives])

        loss_sums = torch.squeeze(self.agg_mi_score(embeddings, positive_examples, all_samples),-1)
        sum_loss = loss_sums.mean()

        return sum_loss


    def forward(self, embeddings, community_pos_options, negative_embs, iter_n, node_similarity_weight=None):
        community_membership_loss = self.network_stucture_loss(embeddings, negative_embs, \
            community_pos_options, iter_n)
        community_membership_loss = Variable(community_membership_loss, requires_grad=True)
        return community_membership_loss
