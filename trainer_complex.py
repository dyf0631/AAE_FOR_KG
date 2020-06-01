from dataset import Dataset
from Model import ConvEncoder, Decoder, ComplEx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
from radam import RAdam
import numpy as np
import scipy.misc
from scipy.misc import imsave
import torch.autograd as autograd

import time


class Trainer:
    def __init__(self, dataset, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.encoder = ConvEncoder(self.dataset.num_ent(), self.dataset.num_rel(), args.emb_dim, self.device)
        self.decoder = Decoder(self.dataset.num_ent(), self.dataset.num_rel() , self.device)
        self.discriminator = ComplEx(self.dataset.num_ent(), self.dataset.num_rel(), args.emb_dim, self.device)
        
        self.args = args
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.BCELoss()


        
    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        print('entity', self.dataset.num_ent(), 'relation', self.dataset.num_rel())
        print('ConvEncoder')
        print('train_simple1')
        print('epoch', self.args.ne)
        print('D_lr',self.args.D_lr)
        print('G_lr',self.args.G_lr)
        print('emb_dim',self.args.emb_dim)
        print('batch_size',self.args.batch_size)
        print('discriminator_range',self.args.discriminator_range)

        entity_onehot = []
        relation_onehot = []

        for i in range(self.dataset.num_ent()):
            onehot = [0 for x in range(self.dataset.num_ent())]
            onehot[i] = 1
            entity_onehot.append(onehot)

        for i in range(self.dataset.num_rel()):
            onehot = [0 for x in range(self.dataset.num_rel())]
            onehot[i] = 1
            relation_onehot.append(onehot)


#********************************admgrad*********************************************************************
        optimizer_D = torch.optim.Adagrad(
            self.discriminator.parameters(),
            lr = self.args.D_lr,
            weight_decay= 0,
            initial_accumulator_value= 0.1)

        optimizer_Encoder = torch.optim.Adagrad(
            self.encoder.parameters(),
            lr = self.args.G_lr,
            weight_decay= 0,
            initial_accumulator_value= 0.1)
        optimizer_Decoder = torch.optim.Adagrad(
            self.decoder.parameters(),
            lr = self.args.G_lr,
            weight_decay= 0,
            initial_accumulator_value= 0.1)

        for epoch in range(1, self.args.ne+1):

            # start_time = time.time()

            last_batch = False
            total_d_loss = 0.0
            total_g_loss = 0.0
            while not last_batch:
                
                pos_batch = self.dataset.next_pos_batch(self.args.batch_size)
                last_batch = self.dataset.was_last_batch()

                h_onehot = []
                r_onehot = []
                t_onehot = []
                for i in pos_batch[:,0]:
                    one_hot = entity_onehot[i]
                    h_onehot.append(one_hot)
                for i in pos_batch[:,2]:
                    one_hot = entity_onehot[i]
                    t_onehot.append(one_hot)
                for i in pos_batch[:,1]:
                    one_hot = relation_onehot[i]
                    r_onehot.append(one_hot)

                h = torch.tensor(h_onehot).float().to(self.device)
                r = torch.tensor(r_onehot).float().to(self.device)
                t = torch.tensor(t_onehot).float().to(self.device)


                # -----------------
                #  Train Generator
                # ----------------
                optimizer_Encoder.zero_grad()
                optimizer_Decoder.zero_grad()

                encoder_batch = np.repeat(np.copy(pos_batch), 1, axis=0)
                for i in range(self.args.batch_size):
                    if np.random.random()<0.5:
                        encoder_batch[i][0] = pos_batch[i][0]
                        encoder_batch[i][1] = 0
                        encoder_batch[i][2] = pos_batch[i][1]
                    else:
                        encoder_batch[i][0] = pos_batch[i][2]
                        encoder_batch[i][1] = 1
                        encoder_batch[i][2] = pos_batch[i][1]

                encoder_h_onehot = []
                encoder_r_onehot = []
                encoder_position = []

                for i in encoder_batch[:,0]:
                    one_hot = entity_onehot[i]
                    encoder_h_onehot.append(one_hot)
                for i in encoder_batch[:,1]:
                    encoder_position.append([i])
                for i in encoder_batch[:,2]:
                    one_hot = relation_onehot[i]
                    encoder_r_onehot.append(one_hot)


                encoder_h = torch.tensor(encoder_h_onehot).float().to(self.device)
                encoder_p = torch.tensor(encoder_position).float().to(self.device)
                encoder_r = torch.tensor(encoder_r_onehot).float().to(self.device)


                fake_tails =self.encoder(encoder_h, encoder_p, encoder_r)
                construction_heads, construction_postions, construction_rels = self.decoder(fake_tails)

                g_loss = self.reconstruction_loss(construction_heads, encoder_h) + self.reconstruction_loss(construction_rels, encoder_r) + self.reconstruction_loss(construction_postions, encoder_p)

                g_loss.backward()
                total_g_loss += g_loss.cpu().item()
                optimizer_Encoder.step() 
                optimizer_Decoder.step()

                neg_batch = np.repeat(np.copy(pos_batch), self.args.neg_ratio, axis=0)
                for _ in range(self.args.discriminator_range):
                    neg_entity = []
                    for i in range(len(neg_batch)):
                        if np.random.random() < 0.5:
                            temp = []
                            temp_h = pos_batch[i][0]
                            temp_p = [0]
                            temp_r = pos_batch[i][1]
                            
                            temp.append(temp_h)
                            temp.append(temp_p)
                            temp.append(temp_r)
                            neg_entity.append(temp)

                        else:
                            temp = []
                            temp_h = pos_batch[i][2]
                            temp_p = [1]
                            temp_r = pos_batch[i][1]
                            
                            temp.append(temp_h)
                            temp.append(temp_p)
                            temp.append(temp_r)
                            neg_entity.append(temp)
                    
                    temp_h_one_hot = []
                    temp_r_one_hot = []
                    temp_p = []

                    for ele in neg_entity:
                        temp_h_one_hot.append(entity_onehot[ele[0]])
                        temp_r_one_hot.append(relation_onehot[ele[2]])
                        temp_p.append(ele[1])

                    temp_h_one_hot = torch.tensor(temp_h_one_hot).float().to(self.device)
                    temp_r_one_hot = torch.tensor(temp_r_one_hot).float().to(self.device)
                    temp_p = torch.tensor(temp_p).float().to(self.device)
                    neg_tails_index = np.argmax(self.encoder(temp_h_one_hot, temp_p, temp_r_one_hot).cpu().data.numpy(), axis=1)

                    for i in range(len(neg_batch)):
                        if neg_entity[i][1] == [0]:
                            neg_batch[i][2] = neg_tails_index[i]
                        elif neg_entity[i][1] == [1]:
                            neg_batch[i][0] = neg_tails_index[i]
                        else:
                            print('GG')
                    neg_batch[:,-1] = -1

                    batch = np.append(pos_batch, neg_batch, axis=0)
                    np.random.shuffle(batch)

                    full_h_onehot = []
                    full_r_onehot = []
                    full_t_onehot = []

                    for i in batch[:,0]:
                        one_hot = entity_onehot[i]
                        full_h_onehot.append(one_hot)

                    for i in batch[:,2]:
                        one_hot = entity_onehot[i]
                        full_t_onehot.append(one_hot)

                    for i in batch[:,1]:
                        one_hot = relation_onehot[i]
                        full_r_onehot.append(one_hot)

                    full_h = torch.tensor(full_h_onehot).float().to(self.device)
                    full_r = torch.tensor(full_r_onehot).float().to(self.device)
                    full_t = torch.tensor(full_t_onehot).float().to(self.device)
                    labels = torch.tensor(batch[:,3]).float().to(self.device)

                    optimizer_D.zero_grad()
                    scores = self.discriminator(full_h, full_r, full_t)
                    d_loss = torch.sum(F.softplus(-labels * scores)) + (self.args.reg_lambda * self.discriminator.l2_loss() / self.dataset.num_batch(self.args.batch_size))
                    d_loss.backward()
                    optimizer_D.step()

                    for p in self.discriminator.parameters():
                        p.data.clamp_(-1, 1)

                    total_d_loss += d_loss.cpu().item()

                # =================== generator training =======================
                optimizer_Encoder.zero_grad()
                fake_tails =self.encoder(encoder_h, encoder_p, encoder_r)
                generator_score = self.discriminator(encoder_h, encoder_r, fake_tails)

                G_loss = -0.2 * torch.mean(torch.log(generator_score + 1e-6))
                
                G_loss.backward()
                optimizer_Encoder.step()


            # finish_time = time.time()

            # with open("train_time_log.log",'a') as f:
            #     f.write(str(epoch)+"    "+str(start_time)+"    "+str(finish_time)+"\n")
                    
            print("Loss in iteration " + str(epoch) + ": " + str(total_d_loss) + "(" + self.dataset.name + ")")
            print("Loss in iteration " + str(epoch) + ": " + str(total_g_loss) + "(" + self.dataset.name + ")")
        
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

            if epoch % 25 == 0:
                print('epoch', epoch, scores)
                print('neg_batch', neg_batch[:,2])


    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + 'complex' + "/" 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.discriminator, directory + str(chkpnt) + ".chkpnt")