import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

class LightFM(GeneralRecommender):
    """LightFM is a hybrid matrix factorization model representing users and items as linear combinations 
    of their content features' latent factors. Without features, it reduces to a standard Matrix Factorization 
    with user and item biases.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightFM, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        
        # Latent representations (embeddings)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # LightFM specifically includes bias terms for both users and items
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)

        self.loss = BPRLoss()

        # weights initialization
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        user_b = self.user_bias(user).squeeze(-1)
        item_b = self.item_bias(item).squeeze(-1)

        # score = dot(user_emb, item_emb) + user_bias + item_bias
        score = torch.mul(user_e, item_e).sum(dim=1) + user_b + item_b
        return score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_score = self.forward(user, pos_item)
        neg_score = self.forward(user, neg_item)

        loss = self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
