PYTHON=/home/v-jiaya/anaconda3/bin/python
PREPROCESS=/home/v-jiaya/RetriveNMT/fairseq-baseline/preprocess.py
TEXT=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/train/
DATA_BIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/
$PYTHON $PREPROCESS --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $DATA_BIN --output-format binary \
    --srcdict /home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/dict.de.txt \
    --tgtdict /home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/data-bin/dict.en.txt \
    --workers 40
