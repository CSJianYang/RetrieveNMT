javac IndexFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-es/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.es -index $TRAIN/index