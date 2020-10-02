nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export -p
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/
MODEL=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/model/test
/home/v-jiaya/anaconda3/bin/python /home/v-jiaya/RetriveNMT/RetriveNMT/train.py $TEXT \
    --source-lang de --target-lang en \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --min-lr 1e-15 --lr 0.1 --lr-scheduler my_inverse_sqrt --warmup-updates 4000  \
    --max-update 100000 --max-epoch 100 --log-interval 100 --log-format "simple"  \
    --save-interval-updates 1 --save-interval 100 --keep-interval-updates 5 --max-tokens 2048 \
    --arch JRC_Acquis_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 1 --ddp-backend=no_c10d 