#!/bin/bash

cluster_name=${cluster_name:-localhost}
dataset=${dataset:-cifar10}

case "$cluster_name" in
    "localhost")
        case "$dataset" in
            "Tiny-ImageNet-200") datadir="/datasets/tiny-imagenet-200" ;;
            "cifar10") datadir="/datasets/cifar10" ;;
            "cifar100") datadir="/datasets/cifar100" ;;
            "fmnist") datadir="/datasets/fmnist" ;;
            "SVHN") datadir="/datasets/SVHN" ;;
            "mnist") datadir="/datasets" ;;
        esac
        ;;
    *)
        echo "Unknown cluster name: $cluster_name"
        exit 1
        ;;
esac

echo "Data directory for dataset '$dataset' on cluster '$cluster_name': $datadir"










