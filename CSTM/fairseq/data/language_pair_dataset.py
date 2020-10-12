# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    src_mask = None
    tgt_mask = None
    if samples[0].get('src_mask', None) is not None:
        src_mask = merge('src_mask', left_pad=left_pad_target)
        src_mask = src_mask.index_select(0, sort_order)
    if samples[0].get('tgt_mask', None) is not None:
        tgt_mask = merge('tgt_mask', left_pad=left_pad_target)
        tgt_mask = tgt_mask.index_select(0, sort_order)


    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def Source_RetrieveSource_RetrieveTarget_Collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, args=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def Source_RetrieveSource_RetrieveTarget_Split(x, l, args=None):
        """
        split source words and retrive words.
        x: Batch * TimeStep
        sentence format(left pad) [pad, pad, pad, [APPEND], [SRC], x1, x2, x3, [TGT], y1, y2, y3, [SEP], [SRC], x1, x2, x3, [TGT], y1, y2, y3, [SEP], [eos]]
        """
        # choose word to drop
        source_sentences = []
        source_lengths = []
        retrieve_source_sentences = [[]for _ in range(args.retrieve_number)]
        retrieve_source_lengths = [[]for _ in range(args.retrieve_number)]
        retrieve_target_sentences = [[]for _ in range(args.retrieve_number)]
        retrieve_target_lengths = [[]for _ in range(args.retrieve_number)]
        for i in range(l.size(0)):
            words = x[i, -l[i]:]
            append_position = (words == args.APPEND_ID).nonzero()
            if len(append_position) == 1:
                append_position = append_position[0][0]
            else:
                append_position = words.size(0)

            if append_position > 0 and append_position < words.size(0):
                source_words, retrieve_words = words[:append_position], words[append_position:]
            elif append_position == 0:
                source_words = x.new([args.UNK_ID])
                retrieve_words = words
            else:
                source_words = words
                retrieve_words = x.new([args.APPEND_ID,args.SRC_ID,args.unk_idx,args.TGT_ID,args.unk_idx,args.SEP_ID])
                # assert words.split(" "+str(args.APPEND_ID)+" ")  .__len__() == 2, "Retrive sequences must contain one [APPEND] tag! " + words + " len: " + str(words.split(" "+str(self.args.APPEND_ID)+" ").__len__()) + " sent len: {}".format(l[i]) + " APPEND_ID: {}".format(self.args.APPEND_ID)

            source_sentences.append(source_words)
            source_lengths.append(len(source_words))

            sep_positions = (retrieve_words == args.SEP_ID).nonzero().tolist()
            src_positions = (retrieve_words == args.SRC_ID).nonzero().tolist()
            tgt_positions = (retrieve_words == args.TGT_ID).nonzero().tolist()
            assert len(sep_positions) == len(src_positions) and len(sep_positions) == len(tgt_positions), "Please make sure [SEP] [SRC] [TGT] have the same number of occurrences"
            for j in range(min(args.retrieve_number,len(sep_positions))):
                retrieve_source_words = retrieve_words[src_positions[j][0]: tgt_positions[j][0]]
                retrieve_target_words = retrieve_words[tgt_positions[j][0]: sep_positions[j][0]]

                retrieve_source_sentences[j].append(retrieve_source_words)
                retrieve_source_lengths[j].append(len(retrieve_source_words))
                retrieve_target_sentences[j].append(retrieve_target_words)
                retrieve_target_lengths[j].append(len(retrieve_target_words))

        # re-construct source input
        l2 = l.new(source_lengths)
        x2 = x.new(l2.size(0), l2.max()).fill_(args.PAD_ID)
        for i in range(l2.size(0)):
            x2[i, -l2[i]:].copy_(source_sentences[i])

        multi_retrieve_source_x = []
        multi_retrieve_target_x = []

        multi_retrieve_source_l = []
        multi_retrieve_target_l = []
        for i in range(min(args.retrieve_number,len(sep_positions))):
            # re-construct retrieve source input
            retrieve_source_l = l.new(retrieve_source_lengths[i])
            retrieve_source_x = x.new(retrieve_source_l.size(0), retrieve_source_l.max()).fill_(args.PAD_ID)
            for j in range(retrieve_source_l.size(0)):
                retrieve_source_x[j, -retrieve_source_l[j]:].copy_(retrieve_source_sentences[i][j])
            multi_retrieve_source_x.append(retrieve_source_x)
            multi_retrieve_source_l.append(retrieve_source_l)

            # re-construct retrieve target input
            retrieve_target_l = l.new(retrieve_target_lengths[i])
            retrieve_target_x = x.new(retrieve_target_l.size(0), retrieve_target_l.max()).fill_(args.PAD_ID)
            for j in range(retrieve_target_l.size(0)):
                retrieve_target_x[j, -retrieve_target_l[j]:].copy_(retrieve_target_sentences[i][j])
            multi_retrieve_target_x.append(retrieve_target_x)
            multi_retrieve_target_l.append(retrieve_target_l)

        return (x2, l2), (multi_retrieve_source_x, multi_retrieve_source_l), (multi_retrieve_target_x, multi_retrieve_target_l)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    (src_tokens, src_lengths), (RetrieveSource_tokens, RetrieveSource_lengths), (RetrieveTarget_tokens, RetrieveTarget_lengths) = Source_RetrieveSource_RetrieveTarget_Split(src_tokens, src_lengths, args=args)



    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': [src_tokens, RetrieveSource_tokens, RetrieveTarget_tokens],
            'src_lengths': [src_lengths, RetrieveSource_lengths, RetrieveTarget_lengths],
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch

def generate_dummy_batch(num_tokens, collate_fn, src_dict, src_len=128, tgt_dict=None, tgt_len=128):
    """Return a dummy batch with a given number of tokens."""
    bsz = num_tokens // max(src_len, tgt_len)
    return collate_fn([
        {
            'id': i,
            'source': src_dict.dummy_sentence(src_len),
            'target': tgt_dict.dummy_sentence(tgt_len) if tgt_dict is not None else None,
            'src_mask': src_dict.dummy_sentence(src_len),
            'tgt_mask': tgt_dict.dummy_sentence(tgt_len) if tgt_dict is not None else None
        }
        for i in range(bsz)
    ])


class LanguagePairDataset(FairseqDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False, args=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_mask=None
        self.tgt_mask=None
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.args=args

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_mask_item = None
        tgt_mask_item = None
        if  self.src_mask != None:
            src_mask_item = self.src_mask[index]
            tgt_mask_item = self.src_mask[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'src_mask': src_mask_item,
            'tgt_mask': tgt_mask_item
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return Source_RetrieveSource_RetrieveTarget_Collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, args=self.args
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.src_dict, src_len, self.tgt_dict, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else self.src_sizes[index]
        #return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if not self.shuffle:
            return indices

        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]

        return indices



    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.src_mask is not None:
            self.src_mask.prefetch(indices)
        if self.tgt_mask is not None:
            self.tgt_mask.prefetch(indices)
