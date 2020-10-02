PYTHON=/home/v-jiaya/anaconda3/bin/python
PREPROCESS=/home/v-jiaya/RetriveNMT/fairseq-baseline/preprocess.py
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/train/
DATA_BIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/en2de/top2/data-bin/
$PYTHON $PREPROCESS --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $DATA_BIN --output-format binary \
    --joined-dictionary \
    --workers 40
