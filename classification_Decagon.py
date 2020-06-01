import torch
from dataset import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import minmax_scale



class Tester:
    def __init__(self, dataset, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discriminator = torch.load(model_path, map_location=self.device)
        self.discriminator.eval()
        self.dataset = dataset


    def test(self):
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

        pr = []
        roc = []
        p1 = []
        p3 = []
        p5 = []

        file = open('./Decagon_cla.txt', 'r')

        for l in file.readlines():
            l = l.strip()
            head, tail, label = l.split('\t')  # 要求源文件的排序位en1 rel en2
            labels = list(map(int, list(label)))

            y_ture = torch.unsqueeze(torch.tensor(labels),0).long().cpu().numpy()
            r_onehot = []
            h_onehot = [entity_onehot[int(head)]] * self.dataset.num_rel()
            t_onehot = [entity_onehot[int(tail)]] * self.dataset.num_rel()


            for i in range(self.dataset.num_rel()):
                one_hot = relation_onehot[i]
                r_onehot.append(one_hot)

            h = torch.tensor(h_onehot).float().to(self.device)
            r = torch.tensor(r_onehot).float().to(self.device)
            t = torch.tensor(t_onehot).float().to(self.device)


            y_prob = self.discriminator(h, r, t)
            y_prob = y_prob.detach().cpu().numpy()
            y_prob_final = minmax_scale(y_prob) / np.sum(minmax_scale(y_prob))
            y_prob_final = np.expand_dims(y_prob_final, axis = 0)


            
            # y_ture = y_ture.detach().cpu().numpy()
            metric = self.metric_report(y_ture, y_prob_final)
            
            pr.append(metric['pr'])
            roc.append(metric['roc'])
            p1.append(metric['p@1'])
            p3.append(metric['p@3'])
            p5.append(metric['p@5'])

        file.close()

        print('pr', np.mean(pr))
        print('roc', np.mean(roc))
        print('p@1', np.mean(p1))
        print('p@3', np.mean(p3))
        print('p@5', np.mean(p5))


    def metric_report(self, y, y_prob):
        rocs = []
        prs = []
        ks = [1, 3, 5]
        pr_score_at_ks = []
        for k in ks:
            pr_at_k = []
            for i in range(y_prob.shape[0]):
                y_prob_index_topk = np.argsort(y_prob[i])[::-1][:k]
                inter = set(y_prob_index_topk) & set(y[i].nonzero()[0])
                pr_ith = len(inter) / k
                pr_at_k.append(pr_ith)
            pr_score_at_k = np.mean(pr_at_k)
            pr_score_at_ks.append(pr_score_at_k)

        # for i in range(y.shape[1]):
        #     if (sum(y[:, i]) < 1):
        #         continue
        roc = roc_auc_score(y[0], y_prob[0])
        rocs.append(roc)

        prauc = average_precision_score(y[0], y_prob[0])
        prs.append(prauc)

        roc_auc = sum(rocs) / len(rocs)
        pr_auc = sum(prs) / len(prs)

        return {'pr': pr_auc,
                'roc': roc_auc,
                'p@1': pr_score_at_ks[0],
                'p@3': pr_score_at_ks[1],
                'p@5': pr_score_at_ks[2]}