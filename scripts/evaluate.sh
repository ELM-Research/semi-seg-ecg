CUDA_VISIBLE_DEVICES=5 bash scripts/test.sh \
-f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
-o ../configs/bench/isp/1over2.yaml \
--model_path ../exps/resnet18/mean_teacher_boundary_unet/ludb/1over2/best-MeanIoU.pth