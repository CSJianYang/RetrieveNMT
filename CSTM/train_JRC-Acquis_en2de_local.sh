nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEXT=/hdfs/msrmt/v-jiaya/pivot/data/multlingual/share-data-bin/
MODEL=/hdfs/msrmt/v-jiaya/pivot/data/multlingual/model/multi-all-pairs-update1
python //hdfs/msrmt/v-jiaya/pivot/MultiSentEncoder/train.py $TEXT/ \
    --task multilingual_translation --lang-pairs en-de,en-fr,de-en,fr-en,en-en,de-de,fr-fr \
    --share-encoders --share-decoders \
    --decoder-langtok --autoencoder \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler my_inverse_sqrt --warmup-updates 4000 \
    --max-update 1000000 --max-epoch 100 --no-progress-bar --log-interval 100 --log-format "simple"  \
    --save-interval-updates 2000 --keep-interval-updates 5 --save-interval 10 --max-tokens 10000 \
    --arch multilingual_transformer --save-dir $MODEL \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 0.095 --min-lr 1e-12 \
    --update-freq 1 \
    --ddp-backend=no_c10d \
    --use-all-emb \