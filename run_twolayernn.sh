python -u train.py \
    --model twolayernn \
    --hidden-dim 192 \
    --epochs 3 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 128 \
    --finetuning True \
    --lr 0.01 | tee twolayernn.log
