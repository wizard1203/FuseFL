



if [ "$model" == "cnn" ]; then
    num_hidden_features=$cnn_hidden_features
elif [ "$model" == "resnet18" ]; then
    num_hidden_features=$res_base_width
fi


echo "type: $type"
echo "gpu: $gpu"
echo "contrastive_n_views: $contrastive_n_views"
echo "contrastive_weight: $contrastive_weight"
echo "dataset: $dataset"
echo "model: $model"
echo "num_hidden_features: $num_hidden_features"
echo "num_users: $num_users"
echo "alpha: $alpha"
echo "local_ep: $local_ep"
echo "lr: $lr"
echo "fedexnn_classifer: $fedexnn_classifer"
echo "fedexnn_adapter: $fedexnn_adapter"
echo "fedexnn_split_num: $fedexnn_split_num"
echo "fedexnn_self_dropout: $fedexnn_self_dropout"
echo "fedexnn_adapter_constrain_beta: $fedexnn_adapter_constrain_beta"

echo "mlp_hidden_features: $mlp_hidden_features"
echo "cnn_hidden_features: $cnn_hidden_features"
echo "num_layers: $num_layers"
echo "last_exp: $last_exp"
echo "res_base_width: $res_base_width"
echo "datadir: $datadir"
echo "EstFeatNorm: $EstFeatNorm"
echo "SaveFeats: $SaveFeats"
echo "TSNE_points: $TSNE_points"
































