javac SearchFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-fr/train/
java SearchFiles -testPath $TRAIN/dev.fr -index $TRAIN/index -search target -topN 20
java SearchFiles -testPath $TRAIN/test.fr -index $TRAIN/index -search target -topN 20
java SearchFiles -testPath $TRAIN/train.fr -index $TRAIN/index -search target -topN 20