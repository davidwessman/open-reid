python examples/softmax_loss.py --lr 0.025 -b 32 -a inception --logs-dir examples/logs/dukemtmc/softmax/inception -d dukemtmc --seed 1 &&
python examples/softmax_loss.py --lr 0.025 -b 32 -a resnet50 --logs-dir examples/logs/dukemtmc/softmax/resnet50 -d dukemtmc --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a inception --logs-dir examples/logs/dukemtmc/triplet/inception -d dukemtmc --seed 1 &&
python examples/triplet_loss.py --lr 0.00005 -b 32 -a resnet50 --logs-dir examples/logs/dukemtmc/triplet/resnet50 -d dukemtmc --seed 1
