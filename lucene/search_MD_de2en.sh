ROOT=/home/v-jiaya/RetrieveNMT/lucene/
WMT_SHELL=$ROOT/search_MD_en_de_wmt.sh
IWSLT_SHELL=$ROOT/search_MD_en_de_iwslt.sh
JRC_SHELL=$ROOT/search_MD_en_de_JRC.sh
Opensubtitles_SHELL=$ROOT/search_MD_en_de_Opensubtitles.sh
bash $WMT_SHELL de target
bash $IWSLT_SHELL de target
bash $JRC_SHELL de target
bash $Opensubtitles_SHELL de target