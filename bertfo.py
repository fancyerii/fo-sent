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

  label_id = label_map[example.label]

  feature = InputFeatures(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=None,
      label=label_id)
  return feature

def getDataset():
    data_path = "./"
    train = sp.get_train_examples(data_path)
    dev = sp.get_dev_examples(data_path)
    test = sp.get_test_examples(data_path)
    from bert.tokenization import FullTokenizer

    tokenizer = FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"))
    train_feas=[]
    for example in train:
        fea=convert_single_example(example.guid, example, sp.get_labels(), max_seq_len, tokenizer)
        train_feas.append(fea)

    dev_feas=[]
    for example in dev:
        fea=convert_single_example(example.guid, example, sp.get_labels(), max_seq_len, tokenizer)
        dev_feas.append(fea)

    test_feas=[]
    for example in test:
        fea=convert_single_example(example.guid, example, sp.get_labels(), max_seq_len, tokenizer)
        test_feas.append(fea)
    return train_feas,dev_feas,test_feas

train_feas,dev_feas,test_feas=getDataset()

train_ids=[fea.input_ids for fea in train_feas]
train_ids=np.array(train_ids)
train_labels=[fea.label for fea in train_feas]
train_labels=np.array(train_labels)

dev_ids = [fea.input_ids for fea in dev_feas]
dev_ids = np.array(dev_ids)
dev_labels = [fea.label for fea in dev_feas]
dev_labels = np.array(dev_labels)


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


bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")
bert.load_stock_weights(l_bert, bert_ckpt_file)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

model.summary()


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

total_epoch_count=20
mc = keras.callbacks.ModelCheckpoint('fo-best.h5', save_best_only=True,
    save_weights_only=True, period=1)
model.fit(x=train_ids, y=train_labels, shuffle=True, batch_size=2,
          epochs=total_epoch_count, validation_data=(dev_ids,dev_labels),
          callbacks=[mc, create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=2,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
          )


#model.save_weights('./sent-fo.h5', overwrite=True)
