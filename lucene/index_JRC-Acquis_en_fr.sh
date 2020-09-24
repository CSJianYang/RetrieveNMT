javac IndexFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-fr/train/
java IndexFiles -source $TRAIN/train.en -target $TRAIN/train.fr -index $TRAIN/index