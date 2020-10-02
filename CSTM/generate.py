import torch

from fairseq import bleu, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module


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
    multi_Rsource_lengths = []
    multi_Rtarget_lengths = []
    multi_Rsource_tokens = []
    multi_Rtarget_tokens = []
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
                #######################################
                single_Rsource_lengths = [sample_id]
                single_Rtarget_lengths = [sample_id]
                single_Rsource_tokens =  [sample_id]
                single_Rtarget_tokens =  [sample_id]
                for layer_id, (Rsource, Rtarget) in enumerate(zip(encoder_outs[0]['multi_selectlayer'][0], encoder_outs[0]['multi_selectlayer'][1])):
                    Rsource_tokens = utils.strip_pad(Rsource['Rsource_tokens'][i, :], tgt_dict.pad()).int().cpu()
                    Rsource_tokens = src_dict.string(Rsource_tokens, args.remove_bpe)
                    single_Rsource_lengths.append(len(Rsource_tokens.split()))  # contain batchsize * seqlen tokens
                    single_Rsource_tokens.append(Rsource_tokens)
                    if Rtarget['Rtarget_tokens'] is not None:
                        Rtarget_tokens = utils.strip_pad(Rtarget['Rtarget_tokens'][i, :], tgt_dict.pad()).int().cpu()
                        Rtarget_tokens = tgt_dict.string(Rtarget_tokens, args.remove_bpe)
                        single_Rtarget_lengths.append(len(Rtarget_tokens.split()))
                        single_Rtarget_tokens.append(Rtarget_tokens)
                multi_Rsource_lengths.append(single_Rsource_lengths)
                multi_Rtarget_lengths.append(single_Rtarget_lengths)
                multi_Rsource_tokens.append(single_Rsource_tokens)
                multi_Rtarget_tokens.append(single_Rtarget_tokens)
                #######################################
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(torch.cat(sample['net_input']['src_tokens'],dim=1)[i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

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

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']



    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    with open(args.output + ".bleu", "a", encoding="utf-8") as w_bleu:
        w_bleu.write(args + "\n")
        w_bleu.write('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    trans_results.sort(key=lambda key: key[0])
    #print(trans_results.__len__())
    print("saving translation result to {}...".format(args.output))
    with open(args.output, "w", encoding="utf-8") as w:
        for item in trans_results:
            w.write("{}\n".format(item[1].replace("<<unk>>", "")))
    multi_Rsource_lengths.sort(key=lambda key: key[0])
    multi_Rtarget_lengths.sort(key=lambda key: key[0])
    multi_Rsource_tokens.sort(key=lambda key: key[0])
    multi_Rtarget_tokens.sort(key=lambda key: key[0])
    total_source_lengths=[0]*(len(multi_Rsource_lengths[-1])-1)
    total_target_lengths=[0]*(len(multi_Rtarget_lengths[-1])-1)
    with open(args.output+".select.{}".format(args.max_sentences), "w", encoding="utf-8") as w_select:
        for i,(Rsource_tokens,Rtarget_tokens,Rsource_lengths,Rtarget_lenghts) in enumerate(zip(multi_Rsource_tokens,multi_Rtarget_tokens,multi_Rsource_lengths,multi_Rtarget_lengths)):
            w_select.write(" ".join([str(item) for item in Rsource_lengths])+"\n")
            total_source_lengths = [total_source_lengths[i-1] + Rsource_lengths[i] for i in range(1, len(Rsource_lengths))]
            total_target_lengths = [total_target_lengths[i-1] + Rtarget_lenghts[i] for i in range(1, len(Rtarget_lenghts))]
    total_source_lengths_ratio = [str(round(float(total_source_lengths[i])/total_source_lengths[0],8)) for i in range(0,len(total_source_lengths))]
    total_target_lengths_ratio = [str(round(float(total_target_lengths[i])/total_target_lengths[0],8)) for i in range(0,len(total_target_lengths))]
    print("Rsource selective tokens: {}".format(total_source_lengths))
    print("Rtarget selective tokens: {}".format(total_target_lengths))
    print("Rsource selective ratio: {}".format(total_source_lengths_ratio))
    print("Rtarget selective ratio: {}".format(total_target_lengths_ratio))



    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
