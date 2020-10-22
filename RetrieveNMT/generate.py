import torch

from fairseq import bleu, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module
import itertools

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset, args=args)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    args.unk_idx = task.src_dict.indices['<unk>']
    args.dict_len = task.src_dict.indices.__len__()
    if '[APPEND]' in task.src_dict.indices.keys():
        args.APPEND_ID = task.src_dict.indices['[APPEND]']
        print("[APPEND] ID: {}".format(args.APPEND_ID))
    else:
        args.APPEND_ID = -1
    if '[SRC]' in task.src_dict.indices.keys():
        args.SRC_ID = task.src_dict.indices['[SRC]']
        print("[SRC] ID: {}".format(args.SRC_ID))
    else:
        args.SRC_ID = -1
    if '[TGT]' in task.src_dict.indices.keys():
        args.TGT_ID = task.src_dict.indices['[TGT]']
        print("[TGT] ID: {}".format(args.TGT_ID))
    else:
        args.TGT_ID = -1
    if '[SEP]' in task.src_dict.indices.keys():
        args.SEP_ID = task.src_dict.indices['[SEP]']
        print("[SEP] ID: {}".format(args.SEP_ID))
    else:
        args.SEP_ID = -1
    if '</s>' in task.src_dict.indices.keys():
        args.EOS_ID = task.src_dict.indices['</s>']
    else:
        args.EOD_ID = -1
    if '<pad>' in task.src_dict.indices.keys():
        args.PAD_ID = task.src_dict.indices['<pad>']
    else:
        args.PAD_ID = -1

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )
    _model_args.avgpen = args.avgpen
    task.datasets['test'].args=_model_args

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    select_retrieve_tokens = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        trans_results = []
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos, encoder_outs = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens, retrieve_source_tokens, retrieve_target_tokens = sample['net_input']['src_tokens']
                retrieve_tokens = list(itertools.chain.from_iterable(zip(retrieve_source_tokens, retrieve_target_tokens)))
                retrieve_tokens = torch.cat(retrieve_tokens, dim=1)
                all_tokens = torch.cat([src_tokens, retrieve_tokens], dim=1)
                src_tokens = utils.strip_pad(all_tokens[i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()


                #

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    trans_results.append((sample_id, hypo_str))
                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))
                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

                    # add select tokens
                    select_retrieve_tokens.append([sample_id,
                                                   src_str,
                                                   target_str,
                                                   sample['predict_ground_truth'][i, :],
                                                   retrieve_tokens[i, :],
                                                   encoder_outs[0]['new_retrieve_tokens'][i, :],
                                                   utils.strip_pad(retrieve_tokens[i, :],
                                                                   src_dict.pad()).tolist(),
                                                   utils.strip_pad(encoder_outs[0]['new_retrieve_tokens'][i, :],
                                                                   src_dict.pad()).tolist()])

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']



    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    trans_results.sort(key=lambda key: key[0])
    print("saving translation result to {}...".format(args.output))
    with open(args.output, "w", encoding="utf-8") as w:
        for item in trans_results:
            w.write("{}\n".format(item[1].replace("<<unk>>", "")))
    select_retrieve_tokens.sort(key=lambda key: key[0])
    orig_retrieve_tokens_length = 0
    select_retrieve_tokens_length = 0
    correct_tokens = 0
    with open(args.output+".select", "w", encoding="utf-8") as w_select:
        for item in select_retrieve_tokens:
            sample_id, src_str, target_str, sample_predict_ground_truth, sample_orig_id, sample_select_retrieve_id, sample_orig_retrieve_tokens, sample_select_retrieve_tokens = item
            retrieve_str = src_dict.string(sample_orig_retrieve_tokens, args.remove_bpe)
            select_str = src_dict.string(sample_select_retrieve_tokens, args.remove_bpe)
            w_select.write("{}\n{}\n{}\n{}\n\n".format(src_str, target_str, retrieve_str, select_str))
            orig_retrieve_tokens_length += len(sample_orig_retrieve_tokens)
            select_retrieve_tokens_length += len(sample_select_retrieve_tokens)
            #calculate accuracy
            correct_tokens += ((sample_select_retrieve_id != _model_args.PAD_ID).long() == sample_predict_ground_truth).masked_fill((sample_orig_id==_model_args.PAD_ID).byte(), 0).sum()

    ratio=select_retrieve_tokens_length/float(orig_retrieve_tokens_length)
    accuracy=correct_tokens.tolist()/float(orig_retrieve_tokens_length)
    print("Selective Tokens: {}".format(ratio))
    print("Correct Tokens: {}".format(accuracy))

    with open("{}.RetrieveNMT.BLEU".format(args.output), "a", encoding="utf-8") as w:
        w.write('{}->{}: Generate {} with beam={} and lenpen={}: {};\tSelection Ratio: {};\tAccuracy\n'.format(args.source_lang, args.target_lang,
                                                                              args.gen_subset, args.beam,
                                                                              args.lenpen, scorer.result_string(),
                                                                              ratio,
                                                                              accuracy))




    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
