TEST_SHELL=/home/v-jiaya/RetrieveNMT/RetrieveNMT/shells/JRC-Acquis/segment/test_JRC.sh
echo "translate en->de"
bash $TEST_SHELL en de
echo "translate de->en"
bash $TEST_SHELL de en
echo "translate en->fr"
bash $TEST_SHELL en fr
echo "translate fr->en"
bash $TEST_SHELL fr en
echo "translate en->es"
bash $TEST_SHELL en es
echo "translate es->en"
bash $TEST_SHELL es en


