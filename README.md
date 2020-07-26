> original README.md of OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py/blob/master/README.md
>
# OpenNMT-py-flat
This is a simple implementation of several experimental funcitons in NMT.
Towards now, the following functions are implemented.

- segment embedding
- flat transformer(unified encoder)
- NFR tag

## segment embedding
In some settings like document-level translation or NFR, input sequence are
consist of two parts seperated by a special token.
Segment embedding assigns extra input stream which consists only 0 and 1 for each
token so that model can recognize whether a token comes from the first part or the
second part.

Default token of the seperation must be "@@@" in current implementation(will be able to change in future).

All tokens before the **first** "@@@" are regarded as the first part. Thus they would be have
a segment token id "0". All the other tokens are "1".
### usage
Add "--segment_embedding" as a training option.

During testing, no extra options are needed.

## Flat-Transformer
Add option "--flat_layers N" as a training option.

N means that Encoder's top N layers are processed to be flat layers.
In this implementation, flat layers indicate a layer only
calculates self attention at the positions corresponds to
**part1 (tokens before first @@@)** of the input sequence. Once N is set to
be 1 or greater, the encoder-decoder attention will also be calculated only
towards the vectors from positions corresponding to part1.

Default N is -1, which means that no flat layers are adopted.


Note. N is a integer and -1 <= N <= --layers(number of encoder layers)

## NFR tag
Add option "--train_nfr_tag FILE" and "--valid_nfr_tag FILE" during preprocessing.

Add option '--nfr_tag_mode \[none \| concat \| add\]' and "--nfr_tag_vec_size D" during training.

Add option '--src_nfr_tag FILE' during testing.

--train_nfr_tag requires a file which has equal lines and tokens
as the -train_src file. Every token in the tag file is 0, 1 or 2 and corresponds
to the token in training source corpus at the same position.

0, 1 and 2 represents for 'S'(source token), 'T'(related token in Fuzzy Match) and 
'R'(unrelated token in Fuzzy Match).

Three modes are supported in --nfr_tag_mode option.
- "none" means no nfr tags should be used. 
- "concat" means that tags will be transferred to a
tag embedding vector which will be concatenated with the token embedding.
Note that in this case, hidden_size = tag_emb_size + token_emb_size **IN ENCODER**. Also, because of the discrepancy
between the encoder's tok_emb_size and the decoder's one which needs no tag_emb so that will be equal to hidden_size,
share embedding is not supported if nfr_tag_mode is set to concat.
- "add" means that tags are also transferred to vector but
the size is equal to the token embedding. And tag embedding will
be directly added to the token embedding.