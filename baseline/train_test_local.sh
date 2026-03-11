name="baseline"
data_dir="/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/data/DenseUAV_data/train"
test_dir="/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/data/DenseUAV_data/test" 

# export the variables in opts.yaml to the environment
eval $(python -c "
import yaml
with open('opts.yaml') as f:
    d = yaml.safe_load(f)
for k, v in d.items():
    if v is not None:
        print(f\"export {k}='{v}'\")
")

# cd baseline
# python train.py \
#   --name Loss_Experiment-CELoss-WeightedSoftTripletLoss_alpha10-KLLoss \
#   --data_dir /home/dmmm/Dataset/DenseUAV/data_2022/train \
#   --batchsize 16 \
#   --num_epochs 120 \
#   --backbone ViTS-224 \
#   --head SingleBranch \
#   --num_worker 8 \
#   --h 224 --w 224 \
#   --rr uav --ra satellite --re satellite \
#   --cls_loss CELoss \
#   --feature_loss WeightedSoftTripletLoss \
#   --kl_loss KLLoss \
#   --droprate 0.5 \
#   --pad 0

# old: --rotate $rotate, --triplet_loss $triplet_loss, --WSTR $WSTR (invalid args)
# new: --rr $rr, --feature_loss $feature_loss, WSTR removed
python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num --rr $rr \
--feature_loss $feature_loss --block $block --lr $lr --num_worker $num_worker --head $head \
--num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from


# cd checkpoints/$name
# python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python evaluate_gpu.py
# python evaluateDistance.py
cd ../../

