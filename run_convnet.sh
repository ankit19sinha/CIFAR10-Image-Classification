python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 300 \
    --epochs 1 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 128 \
    --finetuning False \
    --lr 0.1 | tee convnet.log
