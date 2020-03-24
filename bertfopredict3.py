import bert
import numpy as np
import os
import math
import tensorflow as tf
model_dir = "/home/lili/data/chinese_L-12_H-768_A-12"

bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")


from sent_processor import SentProcessor, convert_examples_to_features
from transformers.data.processors.utils import InputFeatures

sp = SentProcessor()
max_seq_len = 512

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if example.label is not None:
    label_id = label_map[example.label]
  else:
    label_id = None

  feature = InputFeatures(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=None,
      label=label_id)
  return feature

from tensorflow import keras


l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
#l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

# using the default token_type/segment id 0
#output = l_bert([l_input_ids, l_token_type_ids])                              # output: [batch_size, max_seq_len, hidden_size]
output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]


cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
#cls_out = keras.layers.Dropout(0.5)(cls_out)
#logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
#logits = keras.layers.Dropout(0.5)(logits)
logits = keras.layers.Dense(units=3, activation="softmax")(cls_out)
model = keras.Model(inputs=l_input_ids, outputs=logits)
model.build(input_shape=(None, max_seq_len))



model.load_weights('fo-best.h5')


def get_model():
    return model


from bert.tokenization import FullTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
tokenizer = FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"))



