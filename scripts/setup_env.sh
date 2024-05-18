python3=~/anaconda3/envs/py36/bin/python

gpu=1

type=progressive
num_users=5
alpha=0.5
local_ep=20


lr=0.01

type=pretrain
progressive_classifer=fixed
# model=cnn

model=cnn
mlp_hidden_features=100
cnn_hidden_features=128
num_layers=3
num_hidden_features=$cnn_hidden_features


model=resnet18
mlp_hidden_features=100
cnn_hidden_features=128
res_base_width=64
num_hidden_features=$res_base_width


# type=fed-expandable
fedexnn_classifer=avg
fedexnn_adapter=avg
# fedexnn_adapter=cnn1x1

fedexnn_split_num=2
fedexnn_self_dropout=0.0
fedexnn_adapter_constrain_beta=0.0

contrastive_train=False
contrastive_n_views=2
contrastive_weight=1.0

EstFeatNorm=no
last_exp=no
SaveFeats=no
TSNE=no
TSNE_points=500









