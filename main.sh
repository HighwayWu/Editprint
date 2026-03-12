python main.py \
    --test \
    --model 'EditprintFramework' \
    --resume 'weights/editprint_model.pt' \
    --batch_size 8 \
    --batch_aug_num 2 \
    --batch_rep_num 4 \
    --batch_size_test 8 \
    2>&1 | tee out_dir/log.txt