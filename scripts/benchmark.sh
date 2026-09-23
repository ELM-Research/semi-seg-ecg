CUDA_VISIBLE_DEVICES=4,6 bash scripts/train.sh \
-f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
-o ../configs/bench/ludb/1over2.yaml \
--gpus 4,6


CUDA_VISIBLE_DEVICES=4,6 bash scripts/train.sh \
-f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
-o ../configs/bench/ludb/1over4.yaml \
--gpus 4,6


CUDA_VISIBLE_DEVICES=4,6 bash scripts/train.sh \
-f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
-o ../configs/bench/ludb/1over8.yaml \
--gpus 4,6

CUDA_VISIBLE_DEVICES=4,6 bash scripts/train.sh \
-f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
-o ../configs/bench/ludb/1over16.yaml \
--gpus 4,6