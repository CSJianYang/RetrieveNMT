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
        with open(args.new_src_set, "w", encoding="utf-8") as new_src_w:
            print("reading source data file: {}".format(args.src_set))
            src_lines = src_r.readlines()
            for line_id, src_line in tqdm.tqdm(enumerate(src_lines)):
                src_line=src_line.strip()
                concat_lines = src_line.split(" [APPEND] ")[1].split(" [SRC] ")
                for item in concat_lines:
                    src, tgt = item.split(" [TGT] ")
                    new_src_w.write("{} ||| {}\n".format(src, tgt))
                    new_src_w.flush()










