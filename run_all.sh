python examples/softmax_loss.py --lr 0.025 -b 32 -a inception --batch-id $1 --logs-dir examples/logs/$1/softmax/inception -d synthetic --seed 1 &&
python examples/softmax_loss.py --lr 0.025 -b 32 -a resnet50 --batch-id $1 --logs-dir examples/logs/$1/softmax/resnet50 -d synthetic --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a inception --batch-id $1 --logs-dir examples/logs/$1/triplet/inception -d synthetic --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a resnet50 --batch-id $1 --logs-dir examples/logs/$1/triplet/resnet50 -d synthetic --seed 1
