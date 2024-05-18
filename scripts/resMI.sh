python3=~/anaconda3/envs/py36/bin/python

gpu=0

type=progressive
num_users=10
alpha=0.5
local_ep=10

dataset=cifar10
datadir=~/datasets/cifar10


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


lr=0.01
# LRS=(0.01 0.03 0.1)
LRS=(0.01)
num_users=2
local_ep=1


$python3 main.py --main_task=MI --type=$type  --gpu $gpu \
--exp_name ${type}-${dataset}-${model}-nh${num_hidden_features}-c${num_users}-a${alpha}-ep${local_ep}-lr${lr} \
--exp_tool_init_sub_dir your_exp_dir \
--checkpoint weights --resume your_exp_dir \
--split_measure_local_module_num 8 \
--progressive_classifer  ${progressive_classifer} \
--model=$model --mlp_hidden_features=$mlp_hidden_features --cnn_hidden_features $cnn_hidden_features --num_layers $num_layers --res_base_width $res_base_width \
--iid=0 --lr=$lr --MI_cos_lr True \
--dataset=${dataset} --datadir $datadir \
--alpha=$alpha --seed=1 --num_users=${num_users} --local_ep=$local_ep \
--wandb_entity your-entity --project_name FuseFL --enable_wandb False --wandb_offline False \
--wandb_key 'your_key'
















































