PYTHON=/home/v-jiaya/anaconda3/bin/python
PREPROCESS=/home/v-jiaya/RetriveNMT/fairseq-baseline/preprocess.py
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/train/
DATA_BIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-fr/en2fr/top2/data-bin/
mkdir -p $DATA_BIN
$PYTHON $PREPROCESS --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $DATA_BIN --output-format binary \
    --joined-dictionary \
    --workers 40
