export CUDA_VISIBLE_DEVICES=2
MODEL_DIR=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/model/en2fr-simple-update1/
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/data-bin/
/home/v-jiaya/anaconda3/bin/python /home/v-jiaya/pivot/fairseq-baseline/generate.py $TEXT/ --path $MODEL_DIR/checkpoint_best.pt --batch-size 64 --beam 8 --source-lang en --target-lang fr --remove-bpe --lenpen 0.5 --min-len 0 --unkpen 0 --no-repeat-ngram-size 4 --output $TEXT/result.txt


