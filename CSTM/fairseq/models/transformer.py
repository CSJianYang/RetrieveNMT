import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)
import random
import itertools
MAX_SEGMENT_EMBEDDINGS=36

@register_model('transformer')
class TransformerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        #split setting
        parser.add_argument('--word-split', type=str, metavar='D', choices=["Source-Retrive", "Source-Rsource-Rtarget"],
                            help='')
        parser.add_argument('--use-collate', action="store_true")
        parser.add_argument('--use-predictlayer', action='store_true',
                            help='use select layer in encoder')
        parser.add_argument('--predict-loss', action='store_true',
                            help='use predict loss')
        parser.add_argument('--use-splitlayer', action='store_true',
                            help='use select layer in encoder')
        parser.add_argument('--use-copylayer', action='store_true',
                            help='use select layer in encoder')
        parser.add_argument('--gate-method', type=str, metavar='STR',
                            help='')
        parser.add_argument('--reset-position', action='store_true',
                            help='reset position of retrive sentences')
        parser.add_argument('--retrieve-number', type=int, metavar='N',
                            help='')
        parser.add_argument('--noise', type=str, help='')
        parser.add_argument('--training-ratio', type=float, help='')
        parser.add_argument('--scale', type=float, help='')


        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 2048
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 2048

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)

        self.args = args
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.word_split = args.word_split


        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=args.left_pad_source,
            learned=args.encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        #
        self.use_predictlayer = args.use_predictlayer if hasattr(args, "use_predictlayer") else True
        self.loss_weight = Weight(2) if self.use_predictlayer else None
        self.use_splitlayer = args.use_splitlayer
        self.use_copylayer = args.use_copylayer
        self.reset_position = args.reset_position
        self.predict_loss = args.predict_loss if hasattr(args, "use_predictlayer") else True

        self.predictlayers = nn.ModuleList([])
        if self.use_predictlayer:
            self.predictlayers.extend([
                TransformerPredictLayer(args)
                for _ in range(1)
            ])
        #self.splitlayer = TransformerEncoderLayer(args) if self.use_splitlayer else None
        #self.sourcelayer = TransformerEncoderLayer(args) if self.use_splitlayer else None
        self.retrieve_number = args.retrieve_number if hasattr(args, "retrieve_number") else 5
        assert self.retrieve_number * 2 + 1 <= MAX_SEGMENT_EMBEDDINGS, "the number of retrieve sentences is too many !"
        self.retrieve_embed_tokens = Embedding(MAX_SEGMENT_EMBEDDINGS, embed_dim, self.padding_idx)
        if hasattr(args,"training_ratio"):
            self.training_ratio = args.training_ratio
        else:
            self.training_ratio = 0.5



    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        if self.word_split=="Source-RetrieveSource-RetrieveTarget":
            src_tokens,  retrieve_source_tokens_list,  retrieve_target_tokens_list = src_tokens
            src_lengths, retrieve_source_lengths_list, retrieve_target_lengths_list = src_lengths
            assert len(retrieve_source_tokens_list) == len(retrieve_target_tokens_list)
            assert len(retrieve_source_lengths_list) == len(retrieve_target_lengths_list)
            assert src_tokens.size(0) == src_lengths.size(0)
            x = self.embed_scale * self.embed_tokens(src_tokens)

            if self.embed_positions is not None:
                x += self.embed_positions(src_tokens)
            if self.retrieve_embed_tokens is not None:
                x += self.retrieve_embed_tokens(src_tokens.new(src_tokens.size(0), src_tokens.size(1)).fill_(0)) #index 0 represent source segment embeddings

            retrieve_tokens_list = list(itertools.chain.from_iterable(
                zip(retrieve_source_tokens_list, retrieve_target_tokens_list)))
            retrieve_lengths_list = list(itertools.chain.from_iterable(
                zip(retrieve_source_tokens_list, retrieve_target_lengths_list)))
            retrieve_x = []
            for i, (retrieve_tokens, retrieve_lengths) in enumerate(zip(retrieve_tokens_list, retrieve_lengths_list)):
                retrive_emb = self.embed_scale * self.embed_tokens(retrieve_tokens)
                if self.embed_positions is not None:
                    retrive_emb += self.embed_positions(retrieve_tokens)
                if self.retrieve_embed_tokens is not None:
                    retrive_emb += self.retrieve_embed_tokens(src_tokens.new(retrieve_tokens.size(0), retrieve_tokens.size(1)).fill_(i+1))
                retrieve_x.append(retrive_emb)
            retrieve_x = torch.cat(retrieve_x, dim=1)
            retrieve_tokens = torch.cat(retrieve_tokens_list, dim=1)
        else:
            if len(src_tokens) == 1 and len(src_tokens) == 1:
                (src_tokens, src_lengths), (Rsource_tokens, Rsource_lengths), (Rtarget_tokens, Rtarget_lengths) = self.source_Rsource_Rtarget_split(src_tokens, src_lengths)
            elif len(src_tokens) == 3 and len(src_tokens) == 3:
                src_tokens, Rsource_tokens, Rtarget_tokens = src_tokens
                src_lengths, Rsource_lengths, Rtarget_lengths = src_lengths

            x = self.embed_scale * self.embed_tokens(src_tokens)
            x += self.embed_positions(src_tokens)
            if self.retreive_embed_tokens is not None:
                x += self.retrieve_embed_tokens(
                    src_tokens.new(src_tokens.size(0), src_tokens.size(1)).fill_(0))

            Rsource_x = self.embed_scale * self.embed_tokens(Rsource_tokens)
            Rsource_x += self.embed_positions(Rsource_tokens)
            if self.retrive_embed_tokens is not None:
                Rsource_x += self.retrive_embed_tokens(
                    src_tokens.new(Rsource_tokens.size(0), Rsource_tokens.size(1)).fill_(1))

            random_prob = random.random()
            if self.training_ratio > 0 and (random_prob > self.args.training_ratio or not self.training):
                Rtarget_x = self.embed_scale * self.embed_tokens(Rtarget_tokens)
                Rtarget_x += self.embed_positions(Rtarget_tokens)
                if self.retrive_embed_tokens is not None:
                    Rtarget_x += self.retrive_embed_tokens(
                        src_tokens.new(Rtarget_tokens.size(0), Rtarget_tokens.size(1)).fill_(2))
            else:
                Rtarget_tokens = None

        concat_x = torch.cat([x, retrieve_x], dim=1).transpose(0, 1)
        concat_padding_mask = torch.cat([src_tokens, retrieve_tokens], dim=1).eq(self.padding_idx)
        # encoder layers
        encoder_states = []
        for i,layer in enumerate(self.layers):
            concat_x = layer(concat_x, concat_padding_mask)
            encoder_states.append(concat_x)


        predict_save_index = None
        for i, predictlayer in enumerate(self.predictlayers):
            concat_x, predict_save_index, predict_prob, new_retrieve_tokens = predictlayer(concat_x, src_tokens, retrieve_tokens)
            concat_padding_mask = torch.cat([src_tokens, new_retrieve_tokens], dim=1).eq(self.padding_idx)


        return {
            'encoder_out': concat_x,  # T x B x C
            'encoder_padding_mask': concat_padding_mask,  # B x T
            'orig_retrieve_tokens': retrieve_tokens, # B x T
            'new_retrieve_tokens': new_retrieve_tokens, # B x T
            'predict_save_index': predict_save_index,
            'predict_prob': predict_prob,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['orig_retrieve_tokens'] is not None:
            encoder_out['orig_retrieve_tokens'] = \
                encoder_out['orig_retrieve_tokens'].index_select(0, new_order)
        if encoder_out['new_retrieve_tokens'] is not None:
            encoder_out['new_retrieve_tokens'] = \
                encoder_out['new_retrieve_tokens'].index_select(0, new_order)
        if encoder_out['predict_save_index'] is not None:
            encoder_out['predict_save_index'] = \
                encoder_out['predict_save_index'].index_select(0, new_order)
        if encoder_out['predict_prob'] is not None:
            encoder_out['predict_prob'] = \
                encoder_out['predict_prob'].index_select(0, new_order)

        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            '{}.layers.{}.{}.{}'.format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.args=args
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim



        self.max_target_positions = args.max_target_positions
        self.padding_idx = embed_tokens.padding_idx
        args.padding_idx = self.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        self.use_copylayer = args.use_copylayer
        if self.use_copylayer:
            self.copylayer = TransformerCopyLayer(args)
        else:
            self.copylayer = None
        self.adaptive_softmax = None
        self.gate_method = args.gate_method

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm

        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)




    def forward(self, prev_output_tokens, encoder_out=None, incremental_state = None, pretrain=False):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        attn_list = []
        # decoder layers
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out['encoder_out'] is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out['encoder_padding_mask'] is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)



        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)


        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states,
                   'orig_retrieve_tokens': encoder_out['orig_retrieve_tokens'],
                   'predict_prob': encoder_out['predict_prob'],
                   'predict_save_index': encoder_out['predict_save_index']}



    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class TransformerConditionedpredictlayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.incorpor_weight = Linear(self.embed_dim, 1)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, source_x, retrive_x, src_tokens, retrive_tokens, padding_idx):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        src_len = src_tokens.size(1)
        retrive_len = retrive_x.size(1)


        source_attn_padding_mask = src_tokens.eq(padding_idx)
        retrive_attn_padding_mask = retrive_tokens.eq(padding_idx)

        residual = retrive_x
        retrive_x = self.maybe_layer_norm(self.self_attn_layer_norm, retrive_x, before=True)
        retrive_x, _ = self.self_attn(
            query=retrive_x,
            key=retrive_x,
            value=retrive_x,
            key_padding_mask=retrive_attn_padding_mask,
            incremental_state=None,
            need_weights=False,
            attn_mask=retrive_attn_padding_mask,
        )
        retrive_x = F.dropout(retrive_x, p=self.dropout, training=self.training)
        retrive_x = residual + retrive_x
        retrive_x = self.maybe_layer_norm(self.self_attn_layer_norm, retrive_x, after=True)

        attn = None
        residual = retrive_x
        retrive_x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        retrive_x, attn = self.encoder_attn(
            query=retrive_x,
            key=source_x,
            value=source_x,
            key_padding_mask=source_attn_padding_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn),
        )
        retrive_x = F.dropout(retrive_x, p=self.dropout, training=self.training)
        retrive_x = residual + retrive_x
        retrive_x= self.maybe_layer_norm(self.encoder_attn_layer_norm, retrive_x, after=True)

        residual = retrive_x
        retrive_x = self.maybe_layer_norm(self.final_layer_norm, retrive_x, before=True)
        retrive_x = F.relu(self.fc1(retrive_x))
        retrive_x = F.dropout(retrive_x, p=self.relu_dropout, training=self.training)
        retrive_x = self.fc2(retrive_x)
        retrive_x = F.dropout(retrive_x, p=self.dropout, training=self.training)
        retrive_x = residual + retrive_x
        retrive_x = self.maybe_layer_norm(self.final_layer_norm, retrive_x, after=True)
        return retrive_x, attn





    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn



class TransformerPredictLayer(nn.Module):
    """Encoder layer block.

    select which words be saved.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.args=args
        self.self_attn = MultiheadAttention(
                     self.embed_dim, args.encoder_attention_heads,
                     dropout=args.attention_dropout,
                 )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.predict = Linear(self.embed_dim, 2)  #[save, drop]: [0,1] denotes save and [1,0] denotes drop
        self.predict_loss = args.predict_loss if hasattr(args, "predict_loss") else False
        self.padding_idx = args.padding_idx

    def forward(self, concat_x, src_tokens, retrieve_tokens):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        max_src_len = src_tokens.size(1)
        retrieve_padding_mask = retrieve_tokens.eq(self.padding_idx)  # B x T

        concat_tokens = torch.cat([src_tokens, retrieve_tokens], dim=1) # B x T
        concat_padding_mask = concat_tokens.eq(self.padding_idx) # B x T

        residual = concat_x
        concat_x = self.maybe_layer_norm(self.self_attn_layer_norm, concat_x, before=True)
        concat_x, attn_weights = self.self_attn(query=concat_x, key=concat_x, value=concat_x, key_padding_mask=concat_padding_mask)
        concat_x = F.dropout(concat_x, p=self.dropout, training=self.training)
        concat_x = residual + concat_x
        concat_x = self.maybe_layer_norm(self.self_attn_layer_norm, concat_x, after=True)

        residual = concat_x
        concat_x = self.maybe_layer_norm(self.final_layer_norm, concat_x, before=True)
        concat_x = F.relu(self.fc1(concat_x))
        concat_x = F.dropout(concat_x, p=self.relu_dropout, training=self.training)
        concat_x = self.fc2(concat_x)
        concat_x = F.dropout(concat_x, p=self.dropout, training=self.training)
        concat_x = residual + concat_x
        concat_x = self.maybe_layer_norm(self.final_layer_norm, concat_x, after=True)
        x = concat_x[:max_src_len,:,:]  # T1 x B x C
        retrieve_x = concat_x[max_src_len:,:,:] # T2 x B x C

        predict_save_index = None
        if self.predict_loss:
            predict_result = self.predict(retrieve_x)
            # 1 represents selected, 0 represents discarded
            predict_save_index = predict_result.max(dim=2)[1].byte().transpose(0, 1)
            predict_save_index = predict_save_index.masked_fill(~retrieve_padding_mask, 0)  # B x T
            new_retrieve_tokens = retrieve_tokens.masked_fill(~predict_save_index, self.padding_idx)

        return concat_x, predict_save_index, predict_result.transpose(0, 1), new_retrieve_tokens # T x B x c -> B x T x C

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x



class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
                     self.embed_dim, args.encoder_attention_heads,
                     dropout=args.attention_dropout,
                 )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.args=args
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

        self.gate_method = args.gate_method
        if hasattr(args, 'training_ratio'):
            self.training_ratio = args.training_ratio
        else:
            self.training_ratio = 0.5

        if hasattr(args, 'scale'):
            self.scale = args.scale
        else:
            self.scale = 0.1

        if self.gate_method == "project-weight":
            self.incorpor_weights_W=Linear(args.decoder_embed_dim, 1)
            self.incorpor_weights_U=Linear(args.decoder_embed_dim, 1)
            self.target_encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None, tgt_encoder_out=None, tgt_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            if self.gate_method == "project-weight":
                x1, attn1 = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                )
                if (random_prob < self.training_ratio and self.training) or self.training_ratio < 0:
                    x = x1
                else:
                    x2, attn2 = self.target_encoder_attn(
                        query=x,
                        key=tgt_encoder_out,
                        value=tgt_encoder_out,
                        key_padding_mask=tgt_padding_mask,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=(not self.training and self.need_attn),
                    )
                    sigma = self.scale * (torch.sigmoid(self.incorpor_weights_W(x1) + self.incorpor_weights_U(x2)))
                    x = (1-sigma)* x1 + sigma * x2
            else:
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                )


            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn





    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class TransformerCopyLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args):
        super().__init__()
        self.p_gen_linear = nn.Linear(args.decoder_embed_dim, 1)


    def forward(self, x, retrive_tokens, attn_weight_list):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        p_gen = None
        p_gen = self.p_gen_linear(x)
        p_gen = torch.sigmoid(p_gen)
        copy_weight = attn_weight_list.sum(1)
        copy_weight = (1 - p_gen) * copy_weight
        return p_gen, copy_weight



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Weight(in_features, value=0.0):
    m = nn.Parameter(torch.Tensor(in_features))
    nn.init.constant_(m, value)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m

def Weight(in_features, value=0.0):
    m = nn.Parameter(torch.Tensor(in_features))
    nn.init.constant_(m, value)
    # nn.init.constant_(m, 1.0 / in_features)
    # nn.init.constant_(m, 1.0)
    return m


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'JRC_Acquis_transformer')
def JRC_Aquis_transformer(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    #
    args.word_split="Source-RetrieveSource-RetrieveTarget"
    args.use_predictlayer = getattr(args, 'use_predictlayer', False)
    args.predict_loss = getattr(args, 'predict_loss', False)
    args.use_splitlayer = getattr(args, 'use_splitlayer', False)
    args.use_copylayer = getattr(args, 'use_copylayer', False)
    args.gate_method = getattr(args, 'gate_method', 'simple-weight')
    args.reset_position = getattr(args, 'reset_position', True)
    args.retrieve_number = getattr(args, 'retrieve_number', 2)
    args.noise = getattr(args, "noise", None)


@register_model_architecture('transformer', 'iwslt17_small_transformer')
def JRC_Aquis_transformer(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.dropout = getattr(args, 'dropout', 0.3)
    #
    args.word_split= getattr(args, 'word_split', 'Source-Retrive')
    args.use_predictlayer = getattr(args, 'use_predictlayer', False)
    args.predict_loss = getattr(args, 'predict_loss', False)
    args.use_splitlayer = getattr(args, 'use_splitlayer', False)
    args.use_copylayer = getattr(args, 'use_copylayer', False)
    args.gate_method = getattr(args, 'gate_method', 'normal')
    args.reset_position = getattr(args, 'reset_position', False)
    args.retrive_number = getattr(args, 'retrive_number', 2)
    args.noise = getattr(args, "noise", None)
    args.training_ratio = getattr(args, "training_ratio", 0.5)
    args.scale = getattr(args, "scale", 1.0)
    args.vote = getattr(args, "vote", 0)
    args.save_threshold = getattr(args, "save_threshold", -2)
    base_architecture(args)





@register_model_architecture('transformer', 'retrive_transformer')
def retrive_transformer(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    #
    args.word_split= getattr(args, 'word_split', 'Source-RetrieveSource-RetrieveTarget')
    #args.use_predictlayer = getattr(args, 'use_predictlayer', False)
    args.predict_loss = getattr(args, 'predict_loss', False)
    args.use_splitlayer = getattr(args, 'use_splitlayer', False)
    args.use_copylayer = getattr(args, 'use_copylayer', False)
    args.gate_method = getattr(args, 'gate_method', 'simple-weight')
    args.reset_position = getattr(args, 'reset_position', False)
    args.retrive_number = getattr(args, 'retrieve_number', 2)
    args.noise = getattr(args, "noise", None)
    args.training_ratio = getattr(args, "training_ratio", 0.5)
    args.scale = getattr(args, "scale", 1.0)
    args.vote = getattr(args, "vote", 0)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
