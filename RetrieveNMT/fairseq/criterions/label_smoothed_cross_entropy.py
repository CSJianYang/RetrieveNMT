# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch
from fairseq import utils
@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.args = args
        self.softmax_temperature = math.sqrt(args.encoder_embed_dim)
        self.use_predictlayer = self.args.use_predictlayer
        self.min_weight = 0.01
        self.PAD_ID = args.PAD_ID
        self.predict_loss = args.predict_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        if self.predict_loss:
            generate_loss, generate_nll_loss = self.compute_loss(model, net_output, sample, reduce=True)
            predict_loss, predict_nll_loss = self.compute_predict_loss(model, net_output, sample, reduce=True)
            weight1 = 1.0
            weight2 = 0.5
            #weight2 = torch.sigmoid(model.encoder.loss_weight[1]) + 0.01
            loss = weight1 * generate_loss + weight2 * predict_loss
            nll_loss = weight1 * generate_nll_loss + weight2 * predict_nll_loss
        else:
            generate_loss, generate_nll_loss = self.compute_loss(model, net_output, sample, reduce=True)
            weight1 = 1.0
            weight2 = 0.0
            predict_loss = torch.tensor(0)
            predict_nll_loss = torch.tensor(0)
            loss = generate_loss
            nll_loss = generate_nll_loss


        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'generate_loss': utils.item(generate_loss.data) if reduce else generate_loss.data,
            'generate_nll_loss': utils.item(generate_nll_loss.data) if reduce else generate_nll_loss.data,
            'predict_loss': utils.item(predict_loss.data) if reduce else predict_loss.data,
            'predict_nll_loss': utils.item(predict_nll_loss.data) if reduce else predict_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'loss_weight': [utils.item(weight1), utils.item(weight2)]
        }
        return loss, sample_size, logging_output

    def compute_predict_loss(self, model, net_output, sample, reduce=True):
        predict_groudtruth = sample['predict_ground_truth']
        predict_groudtruth = predict_groudtruth.view(-1, 1)

        lprobs = model.get_normalized_probs([net_output[1]['predict_prob']], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        non_pad_mask = net_output[1]['orig_retrieve_tokens'].view(-1, 1).ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=predict_groudtruth)[non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
        loss = nll_loss
        return loss, nll_loss



    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)




        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]



        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'generate_loss': sum(log.get('generate_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'generate_nll_loss': sum(log.get('generate_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'predict_loss': sum(log.get('predict_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'predict_nll_loss': sum(log.get('predict_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size

        }
