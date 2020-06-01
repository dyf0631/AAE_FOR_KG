import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


pi = 3.14159265358979323846
class SimpleE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(SimpleE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent_h_embs   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.ent_t_embs   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.rel_embs     = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)
        self.rel_inv_embs = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)
        
    def l2_loss(self):
        return ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) + (torch.norm(self.rel_embs.weight, p=2) ** 2) + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2
    

    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, num_nodes, num_rels, embed_dim, device):
        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.embed_dim = embed_dim
        self.device = device
        # print(self.entity_channel)
        self.entity_embed_layer = nn.Linear(self.num_nodes, self.embed_dim, bias = True).to(self.device)
        self.relation_embed_layer = nn.Linear(self.num_rels, self.embed_dim, bias = True).to(self.device)
        nn.init.xavier_uniform_(self.entity_embed_layer.weight) # init weights according to [9]
        nn.init.constant_(self.entity_embed_layer.bias, 0.0) 
        nn.init.xavier_uniform_(self.relation_embed_layer.weight) # init weights according to [9]
        nn.init.constant_(self.relation_embed_layer.bias, 0.0) 
        self.attention = nn.TransformerEncoderLayer(d_model=(self.embed_dim + self.embed_dim), nhead=1).to(self.device)
        # print("attention down")
        self.leaky_relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout().to(self.device)
        self.linear = nn.Linear(self.embed_dim + self.embed_dim, self.num_nodes,bias = True).to(self.device)
        nn.init.xavier_uniform_(self.linear.weight.data) # init weights according to [9]
        nn.init.constant_(self.linear.bias.data, 0.0)  


    def forward(self, entity, relation):
        entity_embed = self.entity_embed_layer(entity)
        #entity_embed = self.dropout(entity_embed)
        entity_embed = self.leaky_relu(entity_embed)
        relation_embed = self.relation_embed_layer(relation)
        relation_embed = self.leaky_relu(relation_embed)
        # relation_embed = self.dropout(relation_embed)
        x = torch.cat((entity_embed, relation_embed), 1)

##########################################################################
        x = x.view(-1, 1, (self.embed_dim + self.embed_dim))
        x = self.attention(x)
        x = x.view(-1, (self.embed_dim + self.embed_dim))
        # x = self.batch_normalization(x)
#######################################################################

        x = self.linear(x)
        x = self.leaky_relu(x)
        x = F.gumbel_softmax(x, hard=True)
        return x


class Decoder(nn.Module):
    """docstring for Decoder"""

    def __init__(self, num_nodes, num_rels, device):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.device = device
        # self.entity_embed_dim = entity_embed_dim
        # self.relation_embed_dim = relation_embed_dim
        self.attention = nn.TransformerEncoderLayer(d_model=self.num_nodes, nhead=1).to(self.device)
        self.entity_layer = nn.Linear(in_features=self.num_nodes, out_features=self.num_nodes, bias = True).to(self.device)
        self.relation_layer = nn.Linear(in_features=self.num_nodes, out_features=self.num_rels, bias = True).to(self.device)
        self.position_layer = nn.Linear(in_features=self.num_nodes, out_features=1, bias = True).to(self.device)
        self.leaky_relu = nn.ReLU().to(self.device)
        self.entity_softmax = nn.Softmax(dim=1).to(self.device)
        self.realtion_softmax = nn.Softmax(dim=1).to(self.device)
        
        nn.init.xavier_uniform_(self.entity_layer.weight.data)
        nn.init.xavier_uniform_(self.relation_layer.weight.data)
        nn.init.xavier_uniform_(self.position_layer.weight.data)


    def forward(self, x):
        ######################################################
        x = x.view(-1, 1, self.num_nodes)
        x = self.attention(x)
        x = x.view(-1, self.num_nodes)
        ###########################################################
        entity = self.entity_layer(x)
        entity = self.leaky_relu(entity)
        entity = self.entity_softmax(entity)
        relation = self.relation_layer(x)
        relation = self.leaky_relu(relation)
        realtion = self.realtion_softmax(relation)
        position = self.position_layer(x)
        position = self.leaky_relu(position)
        position = F.sigmoid(position)
        return entity, position, realtion

class model(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(model, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent_h_embs   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.ent_t_embs   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.rel_embs     = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)
        self.rel_inv_embs = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)

        sqrt_size = 3 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)
    

    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

class RotatE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(RotatE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.epsilon = 2.0

        self.ent_embs = nn.Linear(self.num_ent, self.emb_dim*2, bias = True).to(self.device)
        self.rel_embs = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)
        self.gamma = nn.Parameter(torch.Tensor([36.0]), requires_grad=False).to(self.device)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / self.emb_dim]),
            requires_grad=False).to(self.device)

        nn.init.uniform_(tensor=self.ent_embs.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_embs.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def l2_loss(self):
        return self.ent_embs.weight.data.norm(p=3)**3 + self.rel_embs.weight.data.norm(p=3).norm(p=3)**3
    

    def forward(self, heads, rels, tails):
        h = self.ent_embs(heads)
        t = self.ent_embs(tails)
        r = self.rel_embs(rels)

        re_head, im_head = torch.chunk(h, 2, dim=1)
        re_tail, im_tail = torch.chunk(t, 2, dim=1)

        phase_relation = r / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # if mode == 'head-batch':
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #     re_score = re_score - re_head
        #     im_score = im_score - im_head
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=1)
        return score

class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(ComplEx, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.emb_e_real   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.emb_e_img   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.emb_rel_real     = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)
        self.emb_rel_img = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)

        nn.init.xavier_normal_(self.emb_e_real.weight.data)
        nn.init.xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_img.weight.data)

    def l2_loss(self):
        return ((torch.norm(self.emb_e_real.weight, p=2) ** 2) + (torch.norm(self.emb_e_img.weight, p=2) ** 2) + (torch.norm(self.emb_rel_real.weight, p=2) ** 2) + (torch.norm(self.emb_rel_img.weight, p=2) ** 2)) / 2
    

    def forward(self, heads, rels, tails):
        re_head = self.emb_e_real(heads)
        im_head = self.emb_e_img(heads)
        re_relation = self.emb_rel_real(rels)
        im_relation = self.emb_rel_img(rels)
        re_tail = self.emb_e_real(tails)
        im_tail = self.emb_e_img(tails)

        # if mode == 'head-batch':
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #     re_score = re_score - re_head
        #     im_score = im_score - im_head
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 1)
        return score

class ConvE(nn.Module):
    def __init__(self, config):
        super(ConvE, self).__init__()


        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)  # must be 200
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.inp_drop = torch.nn.Dropout(config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(config.hidden_size)
        self.register_parameter('b', Parameter(torch.zeros(self.config.entTotal)))
        self.fc = torch.nn.Linear(10368, config.hidden_size)

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h).view(-1, 1, 10, 20)
        r = self.rel_embeddings(batch_r).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([h, r], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(self.config.batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        if self.config.usegpu:
            one_hot = torch.zeros(self.config.batch_size, self.config.entTotal).scatter_(1, batch_t.cpu(), 1).cuda()
        else:
            one_hot = torch.zeros(self.config.batch_size, self.config.entTotal).scatter_(1, batch_t.cpu(), 1)

        one_hot = ((1.0 - self.config.label_smoothing_epsilon) * one_hot) + (1.0 / one_hot.size(1))
        loss = self.loss(pred, one_hot)

        return loss

class ConvEncoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, num_nodes, num_rels, embed_dim, device):
        super(ConvEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.embed_dim = embed_dim
        self.device = device
        # print(self.entity_channel)
        self.entity_embed_layer = nn.Linear(self.num_nodes+1, self.embed_dim, bias = True).to(self.device) # must be 200
        self.relation_embed_layer = nn.Linear(self.num_rels, self.embed_dim, bias = True).to(self.device)
        self.inp_drop = nn.Dropout(0.2).to(self.device)
        self.hidden_drop = nn.Dropout(0.3).to(self.device)
        self.feature_map_drop = nn.Dropout2d(0.3).to(self.device)

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias= True).to(self.device)
        self.bn0 = nn.BatchNorm2d(1).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.num_nodes).to(self.device)
        self.register_parameter('b', Parameter(torch.zeros(self.num_nodes)))
        self.fc = nn.Linear(10368, self.num_nodes).to(self.device)


        nn.init.xavier_uniform_(self.entity_embed_layer.weight.data) # init weights according to [9]
        nn.init.xavier_uniform_(self.relation_embed_layer.weight.data) # init weights according to [9]

    def forward(self, entity, position, relation):
        entity_input = torch.cat((entity, position), 1)
        entity_embed = self.entity_embed_layer(entity_input).view(-1, 1, 10, 20)
        relation_embed = self.relation_embed_layer(relation).view(-1, 1, 10, 20)
        # relation_embed = self.dropout(relation_embed)
        stacked_inputs = torch.cat([entity_embed, relation_embed], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(512, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.gumbel_softmax(x, hard=True)
        return x

class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.gamma = nn.Parameter(torch.Tensor([24.0]), requires_grad=False)
        self.emb_ent = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.emb_rel = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)

        nn.init.xavier_normal_(self.emb_ent.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)


    def l2_loss(self):
        return ((torch.norm(self.emb_ent.weight, p=2) ** 2) + (torch.norm(self.emb_rel.weight, p=2) ** 2))/ 2
    

    def forward(self, heads, rels, tails):
        h = self.emb_ent(heads)
        r = self.emb_rel(rels)
        t = self.emb_ent(tails)
        score = h + r - t

        score = self.gamma.item() - torch.norm(score, p=1)
        return score

class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(DistMult, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.emb_ent   = nn.Linear(self.num_ent, self.emb_dim, bias = True).to(self.device)
        self.emb_rel = nn.Linear(self.num_rel, self.emb_dim, bias = True).to(self.device)

        nn.init.xavier_normal_(self.emb_ent.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def l2_loss(self):
        return ((torch.norm(self.emb_ent.weight, p=2) ** 2) + (torch.norm(self.emb_rel.weight, p=2) ** 2)) / 2
    

    def forward(self, heads, rels, tails):
        h = self.emb_ent(heads)
        r = self.emb_rel(rels)
        t= self.emb_ent(tails)

        score = (h * r) * t
        score = score.sum(dim = 1)
        return score
