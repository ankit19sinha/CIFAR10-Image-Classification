python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 500 \
    --epochs 3 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 128 \
    --finetuning True \
    --lr 0.1 | tee mymodel.log
