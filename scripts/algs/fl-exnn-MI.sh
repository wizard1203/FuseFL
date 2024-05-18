
type=fed-expandable
source scripts/resetup_env.sh


resume={${last_exp}/weights:-'/your-dir/weights'}


$python3 main.py --main_task=MI --type=$type  --gpu $gpu  --debug $debug \
--exp_name ${type}-${dataset}-${model}-nh${num_hidden_features}-c${num_users}-a${alpha}-ep${local_ep}-lr${lr}-clsf${fedexnn_classifer}-adp${fedexnn_adapter}-nxnn${fedexnn_split_num} \
--EstFeatNorm $EstFeatNorm --SaveFeats $SaveFeats \
--TSNE $TSNE --TSNE_points $TSNE_points \
--exp_tool_init_sub_dir $last_exp \
--checkpoint $checkpoint  --resume  ${last_exp}/weights \
--split_measure_local_module_num 8 \
--fedexnn_classifer  ${fedexnn_classifer} --fedexnn_adapter ${fedexnn_adapter}  --fedexnn_split_num ${fedexnn_split_num} \
--fedexnn_self_dropout $fedexnn_self_dropout --fedexnn_adapter_constrain_beta $fedexnn_adapter_constrain_beta \
--model=$model --mlp_hidden_features=$mlp_hidden_features --cnn_hidden_features $cnn_hidden_features --num_layers $num_layers --res_base_width $res_base_width \
--iid=0 --lr=$lr \
--dataset=${dataset} --datadir $datadir \
--alpha=$alpha --seed=1 --num_users=${num_users} --local_ep=$local_ep \
--wandb_entity your-entity --project_name FuseFL --enable_wandb $enable_wandb --wandb_offline False \
--wandb_key 'your_key'














