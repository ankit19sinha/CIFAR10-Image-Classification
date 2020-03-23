python -u train.py \
    --model softmax \
    --epochs 2 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 128 \
    --finetuning True \
    --lr 0.01 | tee softmax.log