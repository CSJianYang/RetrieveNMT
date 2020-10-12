import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def word_shuffle(x, l, args):
    """
    Randomly shuffle input words.
    x: Batch * TimeStep
    sentence format(left pad) [pad, pad, pad, w1, w2, w3, eos]
    """
    if args.shuffle_prob == 0:
        return x, l

    # define noise word scores
    noise = np.random.uniform(0, args.shuffle_prob, size=(x.size(0), x.size(1)))
    noise[:, -1] = float("inf")  # set [SEP] unchanged
    noise[:, 0] = float("-inf")  # set [APPEND] or [SRC] or [TGT] unchanged
    if x[0, 1] == args.SRC_ID:
        noise[:, 1] = float("0")  # set [APPEND] or [SRC] or [TGT] unchanged
    x2 = x.clone()
    for i in range(l.size(0)):
        # generate a random permutation
        scores = np.arange(l[i].cpu().numpy()) + noise[i, -l[i]:]
        permutation = scores.argsort()
        # shuffle words
        x2[i, -l[i]:].copy_(x2[i, -l[i]:][torch.from_numpy(permutation)])
    return x2, l


def word_dropout(x, l, args):
    """
    Randomly drop input words.
    x: Batch * TimeStep
    sentence format(left pad) [pad, pad, pad, w1, w2, w3, eos]
    """
    if args.drop_prob == 0:
        return x, l
    # choose word to drop
    keep = np.random.rand(x.size(0), x.size(1)) >= args.drop_prob
    keep[:, -1] = True  # do not drop the last sentence symbol
    keep[:, 0] = True  # set [APPEND] or [SRC] or [TGT] unchanged
    if x[0, 1] == args.SRC_ID:
        keep[:, 1] = True  # set [APPEND] or [SRC] or [TGT] unchanged

    sentences = []
    lengths = []
    for i in range(l.size(0)):
        words = x[i, -l[i]:].tolist()
        # randomly drop words from the input
        new_s = [w for j, w in enumerate(words) if keep[i, j + (l.max() - l[i])]]
        # if the wor-dropped sentence only contain the last symbol, save the original sentence
        if len(new_s) == 1:
            new_s = words
        sentences.append(new_s)
        lengths.append(len(new_s))
    # re-construct input
    l2 = torch.LongTensor(lengths)
    x2 = torch.LongTensor(l2.size(0), l2.max()).fill_(args.padding_idx)
    for i in range(l2.size(0)):
        x2[i, -l2[i]:].copy_(torch.LongTensor(sentences[i]))
    return x2, l2


def word_unk(x, l, args):
    """
    Randomly blank input words.
    x: Batch * TimeStep
    sentence format(left pad) [pad, pad, pad, w1, w2, w3, eos]
    """
    if args.unk_prob == 0:
        return x, l

    # define words to blank
    keep = np.random.rand(x.size(0), x.size(1)) >= args.unk_prob
    keep[:, -1] = 1  # do not blank the last sentence symbol
    keep[:, 0] = True  # set [APPEND] or [SRC] or [TGT] unchanged
    if x[0, 1] == args.SRC_ID:
        keep[:, 1] = True  # set [APPEND] or [SRC] or [TGT] unchanged

    sentences = []
    for i in range(l.size(0)):
        words = x[i, -l[i]:].tolist()
        # randomly blank words from the input
        new_s = [w if keep[i, j + (l.max() - l[i])] else args.unk_idx for j, w in enumerate(words)]
        sentences.append(new_s)
    # re-construct input
    x2 = torch.LongTensor(l.size(0), l.max()).fill_(args.padding_idx)
    for i in range(l.size(0)):
        x2[i, -l[i]:].copy_(torch.cuda.LongTensor(sentences[i]))
    return x2, l


def add_noise(src_tokens, src_lengths, args):
    """
    Add noise to the encoder input.
    """
    src_tokens, src_lengths = word_shuffle(src_tokens, src_lengths, args)
    src_tokens, src_lengths = word_dropout(src_tokens, src_lengths, args)
    src_tokens, src_lengths = word_unk(src_tokens, src_lengths, args)
    return src_tokens, src_lengths


def Collate(
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


def Source_RetrieveSource_Collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, args=None, training=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def Source_RetrieveSource_Split(x, l, args=None):
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
                retrieve_words = x.new([args.APPEND_ID])
                # assert words.split(" "+str(args.APPEND_ID)+" ")  .__len__() == 2, "Retrive sequences must contain one [APPEND] tag! " + words + " len: " + str(words.split(" "+str(self.args.APPEND_ID)+" ").__len__()) + " sent len: {}".format(l[i]) + " APPEND_ID: {}".format(self.args.APPEND_ID)

            source_sentences.append(source_words)
            source_lengths.append(len(source_words))

            sep_positions = (retrieve_words == args.SEP_ID).nonzero().tolist()
            src_positions = (retrieve_words == args.SRC_ID).nonzero().tolist()
            assert len(sep_positions) == len(src_positions) and len(sep_positions) == len(tgt_positions), "Please make sure [SEP] [SRC] [TGT] have the same number of occurrences"
            for j in range(len(sep_positions)):
                retrieve_source_words = retrieve_words[src_positions[j][0]: sep_positions[j][0]]
                retrieve_source_sentences[j].append(retrieve_source_words)
                retrieve_source_lengths[j].append(len(retrieve_source_words))


        # re-construct source input
        l2 = l.new(source_lengths)
        x2 = x.new(l2.size(0), l2.max()).fill_(args.PAD_ID)
        for i in range(l2.size(0)):
            x2[i, -l2[i]:].copy_(source_sentences[i])

        multi_retrieve_source_x = []
        multi_retrieve_target_x = []

        multi_retrieve_source_l = []
        multi_retrieve_target_l = []
        for i in range(len(sep_positions)):
            # re-construct retrieve source input
            retrieve_source_l = l.new(retrieve_source_lengths[i])
            retrieve_source_x = x.new(retrieve_source_l.size(0), retrieve_source_l.max()).fill_(args.PAD_ID)
            for j in range(retrieve_source_l.size(0)):
                retrieve_source_x[j, -retrieve_source_l[j]:].copy_(retrieve_source_sentences[i][j])
            multi_retrieve_source_x.append(retrieve_source_x)
            multi_retrieve_source_l.append(retrieve_source_l)


        return (x2, l2), (multi_retrieve_source_x, multi_retrieve_source_l)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    (src_tokens, src_lengths), (RetrieveSource_tokens, RetrieveSource_lengths) = Source_RetrieveSource_Split(src_tokens, src_lengths, args=args)

    if args.noise is not None and training:
        assert len(args.noise.split(",")) == 3, "noise setting must be three probs"
        args.shuffle_prob, args.drop_prob, args.unk_prob = [float(v) for v in args.noise.split(",")]
        src_tokens, src_lengths = add_noise(src_tokens, src_lengths)


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


def Source_RetrieveTarget_Collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, args=None, training=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def Source_RetrieveTarget_Split(x, l, args=None):
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
                retrieve_words = x.new([args.APPEND_ID])
                # assert words.split(" "+str(args.APPEND_ID)+" ")  .__len__() == 2, "Retrive sequences must contain one [APPEND] tag! " + words + " len: " + str(words.split(" "+str(self.args.APPEND_ID)+" ").__len__()) + " sent len: {}".format(l[i]) + " APPEND_ID: {}".format(self.args.APPEND_ID)

            source_sentences.append(source_words)
            source_lengths.append(len(source_words))

            sep_positions = (retrieve_words == args.SEP_ID).nonzero().tolist()
            tgt_positions = (retrieve_words == args.TGT_ID).nonzero().tolist()
            assert len(sep_positions) == len(tgt_positions), "Please make sure [SEP] [SRC] [TGT] have the same number of occurrences"
            for j in range(len(sep_positions)):
                retrieve_source_words = retrieve_words[tgt_positions[j][0]: sep_positions[j][0]]
                retrieve_source_sentences[j].append(retrieve_source_words)
                retrieve_source_lengths[j].append(len(retrieve_source_words))


        # re-construct source input
        l2 = l.new(source_lengths)
        x2 = x.new(l2.size(0), l2.max()).fill_(args.PAD_ID)
        for i in range(l2.size(0)):
            x2[i, -l2[i]:].copy_(source_sentences[i])

        multi_retrieve_source_x = []
        multi_retrieve_target_x = []

        multi_retrieve_source_l = []
        multi_retrieve_target_l = []
        for i in range(len(sep_positions)):
            # re-construct retrieve source input
            retrieve_source_l = l.new(retrieve_source_lengths[i])
            retrieve_source_x = x.new(retrieve_source_l.size(0), retrieve_source_l.max()).fill_(args.PAD_ID)
            for j in range(retrieve_source_l.size(0)):
                retrieve_source_x[j, -retrieve_source_l[j]:].copy_(retrieve_source_sentences[i][j])
            multi_retrieve_source_x.append(retrieve_source_x)
            multi_retrieve_source_l.append(retrieve_source_l)


        return (x2, l2), (multi_retrieve_source_x, multi_retrieve_source_l)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    (src_tokens, src_lengths), (RetrieveSource_tokens, RetrieveSource_lengths) = Source_RetrieveSource_Split(src_tokens, src_lengths, args=args)

    if args.noise is not None and training:
        assert len(args.noise.split(",")) == 3, "noise setting must be three probs"
        args.shuffle_prob, args.drop_prob, args.unk_prob = [float(v) for v in args.noise.split(",")]
        src_tokens, src_lengths = add_noise(src_tokens, src_lengths)


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


def Dynamic_Source_RetrieveSource_RetrieveTarget_Collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, args=None, training=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )


    def calculate_similarity_scores(source_words, retrieve_source_sentences):
        source_words = set(source_words.tolist())
        p_prob = []
        for retrieve_sentence in retrieve_source_sentences:
            retrieve_sentence = set(retrieve_sentence.tolist())
            intersection = source_words & retrieve_sentence
            occurences = len(intersection)
            p_prob.append(occurences)
        p_prob = torch.softmax(p_prob).tolist()
        return p_prob


    def Dynamic_Source_RetrieveSource_RetrieveTarget_Split(x, l, args=None):
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
                retrieve_words = x.new([args.APPEND_ID])
                # assert words.split(" "+str(args.APPEND_ID)+" ")  .__len__() == 2, "Retrive sequences must contain one [APPEND] tag! " + words + " len: " + str(words.split(" "+str(self.args.APPEND_ID)+" ").__len__()) + " sent len: {}".format(l[i]) + " APPEND_ID: {}".format(self.args.APPEND_ID)

            source_sentences.append(source_words)
            source_lengths.append(len(source_words))

            sep_positions = (retrieve_words == args.SEP_ID).nonzero().tolist()
            src_positions = (retrieve_words == args.SRC_ID).nonzero().tolist()
            tgt_positions = (retrieve_words == args.TGT_ID).nonzero().tolist()
            assert len(sep_positions) == len(src_positions) and len(sep_positions) == len(tgt_positions), "Please make sure [SEP] [SRC] [TGT] have the same number of occurrences"
            retrieve_source_words_list = []
            retrieve_target_words_list = []
            for j in range(len(sep_positions)):
                retrieve_source_words = retrieve_words[src_positions[j][0]: tgt_positions[j][0]]
                retrieve_target_words = retrieve_words[tgt_positions[j][0]: sep_positions[j][0]]

                retrieve_source_sentences[j].append(retrieve_source_words)
                retrieve_source_lengths[j].append(len(retrieve_source_words))
                retrieve_target_sentences[j].append(retrieve_target_words)
                retrieve_target_lengths[j].append(len(retrieve_target_words))

            total_retrieve_number = len(sep_positions)
            p_prob = calculate_similarity_scores(source_words, retrieve_source_sentences[i])

        # re-construct source input
        l2 = l.new(source_lengths)
        x2 = x.new(l2.size(0), l2.max()).fill_(args.PAD_ID)
        for i in range(l2.size(0)):
            x2[i, -l2[i]:].copy_(source_sentences[i])

        multi_retrieve_source_x = []
        multi_retrieve_target_x = []

        multi_retrieve_source_l = []
        multi_retrieve_target_l = []
        for i in range(len(sep_positions)):
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
    (src_tokens, src_lengths), (RetrieveSource_tokens, RetrieveSource_lengths), (RetrieveTarget_tokens, RetrieveTarget_lengths) = Dynamic_Source_RetrieveSource_RetrieveTarget_Split(src_tokens, src_lengths, args=args)

    if args.noise is not None and training:
        assert len(args.noise.split(",")) == 3, "noise setting must be three probs"
        args.shuffle_prob, args.drop_prob, args.unk_prob = [float(v) for v in args.noise.split(",")]
        src_tokens, src_lengths = add_noise(src_tokens, src_lengths)


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


def Source_RetrieveSource_RetrieveTarget_Collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, args=None, training=True
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
                retrieve_words = x.new([args.APPEND_ID])
                # assert words.split(" "+str(args.APPEND_ID)+" ")  .__len__() == 2, "Retrive sequences must contain one [APPEND] tag! " + words + " len: " + str(words.split(" "+str(self.args.APPEND_ID)+" ").__len__()) + " sent len: {}".format(l[i]) + " APPEND_ID: {}".format(self.args.APPEND_ID)

            source_sentences.append(source_words)
            source_lengths.append(len(source_words))

            sep_positions = (retrieve_words == args.SEP_ID).nonzero().tolist()
            src_positions = (retrieve_words == args.SRC_ID).nonzero().tolist()
            tgt_positions = (retrieve_words == args.TGT_ID).nonzero().tolist()
            assert len(sep_positions) == len(src_positions) and len(sep_positions) == len(tgt_positions), "Please make sure [SEP] [SRC] [TGT] have the same number of occurrences"
            for j in range(min(len(sep_positions), args.retrieve_number)):
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
        for i in range(min(len(sep_positions), args.retrieve_number)):
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

    if args.noise is not None and training:
        assert len(args.noise.split(",")) == 3, "noise setting must be three probs"
        args.shuffle_prob, args.drop_prob, args.unk_prob = [float(v) for v in args.noise.split(",")]
        src_tokens, src_lengths = add_noise(src_tokens, src_lengths)


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
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False, training=True, **kwargs
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
        self.args = kwargs['args']
        self.training = training

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
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
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        self.args.training = self.training
        # additional monolingual
        if self.args.word_split == "Source-RetrieveSource":
            return Source_RetrieveSource_Collate(
                samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding, args=self.args, training=self.training
            )
        # bilingual
        elif self.args.word_split == "Source-RetrieveSource-RetrieveTarget":
            # x1 x2 x3 [APPEDN] [SRC] x1 x2 x3 [TGT] y1 y2 [SEP] [SRC] x1 x2 x3 [TGT] y1 y2 [SEP]
            #-> [x1 x2 x3] [[x1 x2 x3] [x1 x2 x3]] [[y1 y2] [y1 y2]]
            return Source_RetrieveSource_RetrieveTarget_Collate(
                samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding, args=self.args, training=self.training
            )
        elif self.args.word_split == "Dynamic-Source-RetrieveSource-RetrieveTarget":
            # x1 x2 x3 [APPEDN] [SRC] x1 x2 x3 [TGT] y1 y2 [SEP] [SRC] x1 x2 x3 [TGT] y1 y2 [SEP]
            #-> [x1 x2 x3] [[x1 x2 x3] [x1 x2 x3]] [[y1 y2] [y1 y2]]
            return Dynamic_Source_RetrieveSource_RetrieveTarget_Collate(
                samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding, args=self.args, training=self.training
            )
        else:
             return Collate(
                 samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                 left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                 input_feeding=self.input_feeding
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
