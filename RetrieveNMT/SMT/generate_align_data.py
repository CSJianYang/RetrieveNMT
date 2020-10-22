import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-set', '-src-set', type=str, default=r'/home/v-jiaya/RetrieveNMT/data/MD/en-de/iwslt14-en-de/train/train.en',help='source file')
    parser.add_argument('--new-src-set', '-new-tgt-set', type=str, default=r'/home/v-jiaya/fast_align/data/test.en-de',help='source file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.src_set,"r", encoding="utf-8") as src_r:
        with open(args.tgt_set, "r", encoding="utf-8") as tgt_r:
            with open(args.retrieve_set, "r", encoding="utf-8") as retrieve_r:
                with open(args.dest_set, "w", encoding="utf-8") as concat_w:
                    with open(args.new_tgt_set, "w", encoding="utf-8") as new_tgt_w:
                        print("saving concat results to dest file: {}".format(args.dest_set))
                        print("reading source data file: {}".format(args.src_set))
                        src_lines = src_r.readlines()
                        print("reading target data file: {}".format(args.tgt_set))
                        tgt_lines = tgt_r.readlines()
                        print("reading retrieve data file: {}".format(args.retrieve_set))
                        retrieve_lines = retrieve_r.readlines()
                        if len(src_lines) != len(retrieve_lines):
                            print("src lines and retrieve lines are not equal!")
                            exit(-1)
                        for line_id, (src_line, tgt_line, retrieve_line) in tqdm.tqdm(enumerate(zip(src_lines,tgt_lines,retrieve_lines))):
                            src_line=src_line.strip()
                            tgt_line=tgt_line.strip()
                            if src_line == "" or tgt_line == "":
                                print("{} line: Skipping this empty sentence !".format(line_id+1))
                                continue
                            retrieve_line = retrieve_line.strip()
                            if retrieve_line=="[APPEND] [SRC] [BLANK] [TGT] [BLANK] [SEP]" or retrieve_line.strip()=="":
                                concat_line=src_line.strip()+" [APPEND]" + " [SRC] [BLANK] [TGT] [BLANK] [SEP]" * args.topK
                            elif len(retrieve_line.split(" [SEP] ")) < args.topK:
                                concat_line = "[APPEND] "
                                sents = retrieve_line.replace("[APPEND] ", "").split(" [SEP] ")
                                for i, sent in enumerate(sents):
                                    concat_line += sent.replace(" [SEP]", "") + " [SEP] "
                                pad = " [SRC] [BLANK] [TGT] [BLANK] [SEP]" * (args.topK - len(retrieve_line.split(" [SEP] ")))
                                concat_line = src_line.strip() + " " + concat_line + pad.strip()
                                concat_line = concat_line.replace("  ", " ").strip()
                            else:
                                concat_line = "[APPEND] "
                                sents=retrieve_line.replace("[APPEND] ","").split(" [SEP] ")
                                for i,sent in enumerate(sents):
                                    if i >= args.topK:
                                        break
                                    concat_line += sent.replace(" [SEP]","")+" [SEP] "
                                concat_line = src_line.strip() + " " + concat_line.strip()
                                concat_line = concat_line.replace("  "," ").strip()
                            concat_line += "\n"
                            retrieve_sents = concat_line.split(' [APPEND] ')[1].split(" [SEP] ")
                            assert len(retrieve_sents) == args.topK, concat_line
                            assert concat_line.count("[SRC]") == args.topK, concat_line
                            assert concat_line.count("[TGT]") == args.topK, concat_line
                            assert concat_line.count("[SEP]") == args.topK, concat_line
                            assert concat_line.count("[APPEND]") == 1, concat_line


                            concat_w.write(concat_line)
                            concat_w.flush()
                            new_tgt_w.write("{}\n".format(tgt_line))
                            new_tgt_w.flush()










