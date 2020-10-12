export CUDA_VISIBLE_DEVICES=0
MODEL_DIR=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/model/de2en-simple-update1/
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/
/home/v-jiaya/anaconda3/bin/python /home/v-jiaya/pivot/fairseq-baseline/generate.py $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 8 --source-lang de --target-lang en --remove-bpe --lenpen 0.5 --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt


