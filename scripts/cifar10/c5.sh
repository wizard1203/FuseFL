cluster_name=localhost
dataset=cifar10
# dataset=SVHN
# dataset=fmnist
# dataset=mnist
# dataset=cifar100

source scripts/setup_env.sh
source scripts/path.sh

gpu=3

debug=False
enable_wandb=True

# local_ep=1

# num_users=5

# alpha=0.5
# local_ep=200

# checkpoint=weights
# res_base_width=64
# source scripts/algs/ensemble-train.sh


# checkpoint=weights
# fedexnn_adapter=cnn1x1
# res_base_width=20
# fedexnn_split_num=8
# local_ep=50
# source scripts/algs/fl-exnn.sh


# checkpoint=no
# fedexnn_adapter=cnn1x1
# res_base_width=20
# fedexnn_split_num=4
# local_ep=50
# source scripts/algs/fl-exnn.sh

# checkpoint=no
# fedexnn_adapter=cnn1x1
# res_base_width=20
# fedexnn_split_num=2
# local_ep=100
# source scripts/algs/fl-exnn.sh



num_users=5

alpha=0.5
local_ep=200

checkpoint=weights
res_base_width=64
# source scripts/algs/ensemble-train.sh


# checkpoint=no
# fedexnn_adapter=cnn1x1
# res_base_width=14
# fedexnn_split_num=4
# local_ep=50
# source scripts/algs/fl-exnn.sh

# checkpoint=no
# fedexnn_adapter=cnn1x1
# res_base_width=14
# fedexnn_split_num=2
# local_ep=100
# source scripts/algs/fl-exnn.sh


# checkpoint=weights
# fedexnn_adapter=cnn1x1
# res_base_width=14
# fedexnn_split_num=8
# local_ep=50
# source scripts/algs/fl-exnn.sh


checkpoint=no
fedexnn_adapter=cnn1x1
res_base_width=20
fedexnn_split_num=4
local_ep=50
source scripts/algs/fl-exnn.sh

checkpoint=no
fedexnn_adapter=cnn1x1
res_base_width=20
fedexnn_split_num=2
local_ep=100
source scripts/algs/fl-exnn.sh


checkpoint=no
fedexnn_adapter=cnn1x1
res_base_width=20
fedexnn_split_num=8
local_ep=50
source scripts/algs/fl-exnn.sh










