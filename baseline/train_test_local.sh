name="baseline"
data_dir="/home/dmmm/Dataset/DenseUAV/data_2022/train"
test_dir="/home/dmmm/Dataset/DenseUAV/data_2022/test" 

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num --rotate $rotate \
--triplet_loss $triplet_loss --block $block --WSTR $WSTR --lr $lr --num_worker $num_worker --head $head \
--num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from


cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python evaluate_gpu.py
python evaluateDistance.py
cd ../../

