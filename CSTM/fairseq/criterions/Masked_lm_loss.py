# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('bert_loss')
class BertCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        encoder_loss = None
        decoder_loss = None
        if self.args.pretrain_encoder and self.args.pretrain_decoder:
            encoder_loss, _ = self.compute_loss(model, net_output[1]['encoder_out_pred'], sample, reduce=reduce)
            decoder_loss, _ = self.compute_loss(model, net_output[0], sample, reduce=reduce)
            loss = encoder_loss + decoder_loss
        elif self.args.pretrain_encoder and not self.args.pretrain_decoder:
            encoder_loss, _ = self.compute_loss(model, net_output[1]['encoder_out_pred'], sample, reduce=reduce)
            loss = encoder_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        if self.args.src_mask and sample['src_mask'] is not None:
            sample_size = int(sample['src_mask'].sum().cpu().numpy())
        if self.args.tgt_mask and sample['tgt_mask'] is not None:
            sample_size = int(sample['tgt_mask'].sum().cpu().numpy())


        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'encoder_loss':float(encoder_loss.detach().cpu().numpy()) if encoder_loss is not None else None,
            'decoder_loss':float(decoder_loss.detach().cpu().numpy()) if decoder_loss is not None else None
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        #lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = model.get_masked_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        if sample['src_mask'] is not None:
            mask = model.get_src_masks(sample).view(-1)
            loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=False)
            mask_loss = loss * mask.cuda().float()
            total_loss = mask_loss.sum()
        else:
            loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                              reduce=True)

            total_loss = loss
        return total_loss, total_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        encoder_loss_sum = 0
        decoder_loss_sum = 0
        if logging_outputs[0]['encoder_loss'] is not None:
            encoder_loss_sum = sum(log.get('encoder_loss', 0) for log in logging_outputs)
            encoder_loss_sum = encoder_loss_sum / sample_size / math.log(2)
        if logging_outputs[0]['decoder_loss'] is not None:
            decoder_loss_sum = sum(log.get('decoder_loss', 0) for log in logging_outputs)
            decoder_loss_sum = decoder_loss_sum / sample_size / math.log(2)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'encoder_loss': encoder_loss_sum,
            'decoder_loss': decoder_loss_sum,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / sample_size / math.log(2)
        return agg_output
