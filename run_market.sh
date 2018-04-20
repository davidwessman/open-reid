python examples/softmax_loss.py --lr 0.025 -b 32 -a inception --logs-dir examples/logs/market1501/softmax/inception -d market1501 --seed 1 &&
python examples/softmax_loss.py --lr 0.025 -b 32 -a resnet50 --logs-dir examples/logs/market1501/softmax/resnet50 -d market1501 --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a inception --logs-dir examples/logs/market1501/triplet/inception -d market1501 --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a resnet50 --logs-dir examples/logs/market1501/triplet/resnet50 -d market1501 --seed 1
