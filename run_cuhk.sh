python examples/softmax_loss.py --lr 0.025 -b 32 -a inception --logs-dir examples/logs/cuhk03/softmax/inception -d cuhk03 --seed 1 &&
python examples/softmax_loss.py --lr 0.025 -b 32 -a resnet50 --logs-dir examples/logs/cuhk03/softmax/resnet50 -d cuhk03 --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a inception --logs-dir examples/logs/cuhk03/triplet/inception -d cuhk03 --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a resnet50 --logs-dir examples/logs/cuhk03/triplet/resnet50 -d cuhk03 --seed 1
