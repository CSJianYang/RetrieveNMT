javac IndexFiles.java
TRAIN=/home/v-jiaya/RetrieveNMT/data/MD/en-de/iwslt14-en-de/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.de -index $TRAIN/index
TRAIN=/home/v-jiaya/RetrieveNMT/data/MD/en-de/JRC-Acquis/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.de -index $TRAIN/index
TRAIN=/home/v-jiaya/RetrieveNMT/data/MD/en-de/OpenSubtitles/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.de -index $TRAIN/index
TRAIN=/home/v-jiaya/RetrieveNMT/data/MD/en-de/wmt14-en-de/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.de -index $TRAIN/index