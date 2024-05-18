
type=pretrain

source scripts/resetup_env.sh

$python3 main.py --main_task=LinearProbe --type=$type  --gpu $gpu --debug $debug \
--exp_name ${type}-${dataset}-${model}-nh${num_hidden_features}-c${num_users}-a${alpha}-ep${local_ep}-lr${lr} \
--exp_tool_init_sub_dir $last_exp \
--checkpoint $checkpoint  --resume ${last_exp}/weights \
--split_measure_local_module_num 8 \
--progressive_classifer  ${progressive_classifer} \
--model=$model --mlp_hidden_features=$mlp_hidden_features --cnn_hidden_features $cnn_hidden_features --num_layers $num_layers --res_base_width $res_base_width \
--iid=0 --lr=$lr --MI_cos_lr True \
--dataset=${dataset} --datadir $datadir \
--alpha=$alpha --seed=1 --num_users=${num_users} --local_ep=$local_ep \
--wandb_entity your-entity --project_name FuseFL --enable_wandb $enable_wandb --wandb_offline False \
--wandb_key 'your_key'




















