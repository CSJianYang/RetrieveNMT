export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/v-jiaya/anaconda3/bin/python
PREPROCESS=/home/v-jiaya/RetriveNMT/fairseq-baseline/preprocess.py
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-es/en2es/top2/train/
DATA_BIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-es/en2es/top2/data-bin/
mkdir -p $DATA_BIN
$PYTHON $PREPROCESS --source-lang en --target-lang es \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $DATA_BIN --output-format binary \
    --joined-dictionary \
    --workers 40
