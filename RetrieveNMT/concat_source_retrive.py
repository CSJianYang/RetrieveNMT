from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import tqdm
import argparse
from whoosh.index import open_dir
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-set', '-src-set', type=str, default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/train.de',help='source file')
    parser.add_argument('--retrive-set', '-retrive-set', type=str, default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/train.de.retrive.20',help='target directory')
    parser.add_argument('--dest-set', '-dest-set', type=str,
                        default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/RetriveNMT/en-de/de2en/top2/train/train.de',
                        help='dest file')
    parser.add_argument('--topK', '-topK', type=int, default=2, help='target directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.src_set,"r", encoding="utf-8") as src_r:
        with open(args.retrive_set, "r", encoding="utf-8") as retrive_r:
            with open(args.dest_set, "w", encoding="utf-8") as concat_w:
                print("saving concat results to dest file: {}".format(args.dest_set))
                print("reading source data file: {}".format(args.dest_set))
                src_lines = src_r.readlines()
                print("reading retrive data file: {}".format(args.retrive_set))
                retrive_lines = retrive_r.readlines()
                if len(src_lines) != len(retrive_lines):
                    print("src lines and retrive lines are not equal!")
                    exit(-1)
                for src_line, retrive_line in tqdm.tqdm(zip(src_lines,retrive_lines)):
                    src_line=src_line.strip()
                    retrive_line = retrive_line.strip()
                    if retrive_line=="[APPEND] [SRC] [BLANK] [TGT] [BLANK] [SEP]" or retrive_line.strip()=="":
                        concat_line=src_line.strip()+" [APPEND]" + " [SRC] [BLANK] [TGT] [BLANK] [SEP]" * args.topK
                    elif len(retrive_line.split(" [SEP] ")) < args.topK:
                        concat_line = "[APPEND] "
                        sents = retrive_line.replace("[APPEND] ", "").split(" [SEP] ")
                        for i, sent in enumerate(sents):
                            concat_line += sent.replace(" [SEP]", "") + " [SEP] "
                        pad = " [SRC] [BLANK] [TGT] [BLANK] [SEP]" * (args.topK - len(retrive_line.split(" [SEP] ")))
                        concat_line = src_line.strip() + " " + concat_line + pad.strip()
                        concat_line = concat_line.replace("  ", " ").strip()
                    else:
                        concat_line = "[APPEND] "
                        sents=retrive_line.replace("[APPEND] ","").split(" [SEP] ")
                        for i,sent in enumerate(sents):
                            if i >= args.topK:
                                break
                            concat_line += sent.replace(" [SEP]","")+" [SEP] "
                        concat_line = src_line.strip() + " " + concat_line.strip()
                        concat_line = concat_line.replace("  "," ").strip()
                    concat_line += "\n"
                    retrive_sents = concat_line.split(' [APPEND] ')[1].split(" [SEP] ")
                    if len(retrive_sents) != args.topK:
                        print(concat_line)
                    else:
                        for sent in sents:
                            if len(sent.split(' [TGT] ')) != 2:
                                print(concat_line)
                    if len(concat_line.split(' [APPEND] ')) != 2:
                        print(concat_line)

                    concat_w.write(concat_line)
                    concat_w.flush()










