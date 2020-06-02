# AAE_FOR_KG

Using Autoencoders for Knowledge Graph Embedding

**Introduction**

This is the Pytorch implementaion of some knowledge graph embedding models based on adversarial autoencoders. And we have test these models with drug-drug interaction datasets.

**Dependencies**

This project uses Python 3.5.3, with the following lib dependencies:

* Pytorch 1.3
- Numpy 1.15.1
* Sklearn 0.21

**Usage**

1. If you need to test your own dataset, please put it in `./dataset` directory and format it according to our template.
2. Then you can run commend as follows to train/test/valid the models. For example, this command train a RotatE model on Deepddi dataset.

```python3 -u main_rotate.py -ne 1000 -D_lr 0.5 -G_lr 0.001 -reg 0.3 -dataset Deepddi -emb_dim 200 -neg_ratio 1 -batch_size 512 -save_each 100 -discriminator_range 1```


