#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    # wei 20200809
    nfr_tag_shards = split_corpus(opt.src_nfr_tag, opt.shard_size)
    flat_tag_shards = split_corpus(opt.src_flat_tag, opt.shard_size)
    # end wei
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    # shard_pairs = zip(src_shards, tgt_shards)    # 20200809 wei
    shard_pairs = zip(src_shards, nfr_tag_shards, flat_tag_shards, tgt_shards)

    # for i, (src_shard, tgt_shard) in enumerate(shard_pairs):    # 20200809
    for i, (src_shard, nfr_tag_shard, flat_tag_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            # wei 20200809
            nfr_tag=nfr_tag_shard,
            flat_tag=flat_tag_shard,
            # end wei
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
