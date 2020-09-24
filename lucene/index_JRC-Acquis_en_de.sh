javac IndexFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.de -index $TRAIN/index