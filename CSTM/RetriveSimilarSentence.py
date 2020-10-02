from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import tqdm
import argparse
from whoosh.index import open_dir
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src-set', '-dataset-src-set', type=str, default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/train.de',help='source file')
    parser.add_argument('--data-tgt-set', '-dataset-tgt-set', type=str,
                        default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/train.en',
                        help='source file')
    parser.add_argument('--test-set', '-test-set', type=str, default=r'/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/train/test.de',help='target directory')
    parser.add_argument('--top-K', '-top-K', type=int, default=2, help='target directory')
    parser.add_argument('--indexdir', '-indexdir', type=str, default="/home/v-jiaya/RetriveNMT/data/JRC-Acquis/baseline/train/en-de/indexdir/", help='target directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.data_src_set,"r", encoding="utf-8") as data_src_r:
        with open(args.data_tgt_set, "r", encoding="utf-8") as data_tgt_r:
            with open(args.test_set, "r", encoding="utf-8") as test_r:
                with open(args.test_set+".retrive.top-{}".format(args.top_K), "w", encoding="utf-8") as test_w:
                    schema = Schema(source_tok=TEXT(stored=True), target_tok=TEXT(stored=True), source=TEXT(stored=True), target=TEXT(stored=True))
                    if os.path.exists(args.indexdir):
                        print("Loading index dir {}".format(args.indexdir))
                        ix = open_dir(args.indexdir)
                    else:
                        print("reading source data file")
                        data_src_lines = data_src_r.readlines()
                        print("reading target data file")
                        data_tgt_lines = data_tgt_r.readlines()
                        os.mkdir(args.indexdir)
                        print("Crating index dir {}".format(args.indexdir))
                        ix = create_in(args.indexdir, schema)
                        writer = ix.writer()
                        print("Index Build Start")
                        for data_src_line,data_tgt_line in tqdm.tqdm(zip(data_src_lines,data_tgt_lines)):
                            data_src_line_tok = data_src_line.strip().replace("@@ ","")
                            data_tgt_line_tok = data_tgt_line.strip().replace("@@ ", "")
                            writer.add_document(source_tok=data_src_line_tok, target_tok=data_tgt_line_tok, source=data_src_line.strip(), target=data_tgt_line.strip())
                        writer.commit()
                        print("Index Build Complete")
                    test_lines = test_r.readlines()
                    with ix.searcher() as searcher:
                        for test_line in tqdm.tqdm(test_lines):
                            #test_line = test_line.replace("@@ ","").strip()
                            query = QueryParser("source", ix.schema).parse(test_line)
                            results = searcher.search(query, limit=args.top_K)
                            print("search result number: {}".format(len(results)))
                            if len(results)!=0:
                                for i,result in enumerate(results):
                                    if i == len(result)-1:
                                        test_w.write("[SRC] {} [TGT] {} [SEP]\n".format(result["source"].strip(), result["target"].strip()))
                                    else:
                                        test_w.write("[SRC] {} [TGT] {} [SEP]\t".format(result["source"], result["target"]))
                            else:
                                test_w.write("[SRC] [BLANK] [TGT] [BLANK] [SEP]\n")





