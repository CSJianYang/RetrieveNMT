javac SearchFiles.java
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/test.de -index $TRAIN/index -search target -topN 20
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/test.de -index $TRAIN/index -search target -topN 20
TRAIN=/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/
java SearchFiles -testPath $TRAIN/train.de -index $TRAIN/index -search target -topN 20