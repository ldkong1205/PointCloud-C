CUDA_VISIBLE_DEVICES=3 python test.py \
    --eval True \
    --exp_name robustnesstest_GDANet \
    --model_type insiou \
    --test_batch_size 16