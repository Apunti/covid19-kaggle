# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import io
import os


import biobert.modeling as modeling
import biobert.tokenization as tokenization
import tensorflow.compat.v1 as tf

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.utils.extmath import safe_sparse_dot

#------------------------------------------------------------------------------------------------------

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

#------------------------------------------------------------------------------------------------------

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

#------------------------------------------------------------------------------------------------------

def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
                            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                            "input_ids":
                tf.constant(all_input_ids, 
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
                            "input_mask":
                tf.constant(all_input_mask,
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
                           "input_type_ids":
                tf.constant(all_input_type_ids,
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
             })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn

#------------------------------------------------------------------------------------------------------

def model_fn_builder(bert_config, 
                     init_checkpoint, 
                     layer_indexes, 
                     use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(config=bert_config,
                                   is_training=False,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=input_type_ids,
                                   use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
              raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, 
                                                                                                   init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()            
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {"unique_id": unique_ids}
        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, 
                                                      predictions=predictions, 
                                                      scaffold_fn=scaffold_fn)
        return output_spec
    

    return model_fn

#------------------------------------------------------------------------------------------------------

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

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
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(InputFeatures(unique_id=example.unique_id,
                                      tokens=tokens,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      input_type_ids=input_type_ids))
    return features

#------------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------------

def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
                
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(InputExample(unique_id=unique_id,
                                         text_a=text_a, 
                                         text_b=text_b))
            unique_id += 1
    return examples

#------------------------------------------------------------------------------------------------------

def read_examples_string(input_text):#(input_file, input_text):
    """Read a list of `InputExample`s from an input text."""
    examples = []
    unique_id = 0
    
    with io.StringIO(input_text) as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break

            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)

            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)

            examples.append(InputExample(unique_id=unique_id,
                                         text_a=text_a, 
                                         text_b=text_b))
            unique_id += 1
    return examples

#------------------------------------------------------------------------------------------------------
#               UPDATES BY EKATERINA KRAVCHENKO
#------------------------------------------------------------------------------------------------------
class FeatureExtractor():
    
    def __init__(self,
                 bert_config_file,
                 init_checkpoint,
                 vocab_file,
                 batch_size = 32, # Batch size for predictions
                 max_seq_length = 128,
                 verbose=0):
        
        # configuration
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        layer_indexes = [-1, -2, -3, -4]
        
        if verbose==0: 
            tf.logging.set_verbosity(tf.logging.ERROR)
        elif verbose==1: 
            tf.logging.set_verbosity(tf.logging.INFO)            
        elif verbose==2: 
            tf.logging.set_verbosity(tf.logging.DEBUG)            
        else: 
            raise("Unknown verbosity type.")
   
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, 
                                                    do_lower_case=True)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(master=None,
                                              tpu_config=tf.contrib.tpu.TPUConfig(
                                                                        num_shards=8,
                                                                        per_host_input_for_training=is_per_host))
        # model initialization
        model_fn = model_fn_builder(bert_config=bert_config,
                                    init_checkpoint=init_checkpoint,
                                    layer_indexes=layer_indexes,
                                    use_tpu=False,
                                    use_one_hot_embeddings=False)

        self.estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False,
                                                     model_fn=model_fn,
                                                     config=run_config,
                                                     predict_batch_size=self.batch_size)
        
    def prepare_features(self, input_data, is_file=False):
        # prepare the examples
        if is_file==True:
            examples = read_examples(input_data)
        elif is_file==False:
            examples = read_examples_string(input_data)
        else:
            raise("Need tospecify 'is_file' boolean flag")
            
        features = convert_examples_to_features(examples=examples, 
                                                seq_length=self.max_seq_length, 
                                                tokenizer=self.tokenizer)
        input_fn = input_fn_builder(features=features, seq_length=self.max_seq_length)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature
        
        return unique_id_to_feature, input_fn
        
    def prepare_embedding_csv(self, input_data, csv_filename=None, is_file=False):
        unique_id_to_feature, input_fn = self.prepare_features(input_data, is_file)
        
        # go through examples and process them
        embedding_arr = []        
        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            feature = unique_id_to_feature[int(result["unique_id"])]
            embedding_vector = self.get_embeddings_from_results(result,feature)
            embedding_arr.append(embedding_vector)

        df = pd.DataFrame(embedding_arr, 
                          index=range(len(embedding_arr)),
                          columns=['id'+str(idx) for idx in range(len(embedding_arr[0]))] )
        
        if csv_filename is not None:
            df.to_csv(csv_filename)
            
        return df

    
    def get_embeddings_from_results(self, result, feature):
        embedding_vector = np.zeros(768).tolist()          
        number_of_tokens = len(feature.tokens) - 2

        for i in range(1, number_of_tokens+1):
            feature_vector = np.array([round(float(x), 6) for x in result["layer_output_0"][i:(i + 1)].flat ])
            embedding_vector = embedding_vector + feature_vector

        return np.divide(embedding_vector, number_of_tokens)
    
    
    def get_closest_sentence(self, query_emb_init, paper_id, text, topk=10, encodings_dict = None):  
        """
        Returns a list of tuples (paragraph_id, paragraph_score) for "topk" paragraphs with the highest score.
        """
        if encodings_dict is None:
            csv_filepath = os.path.join("Data/BERT_encodings", paper_id + '.csv')
            df_article = pd.read_csv(csv_filepath, index_col=0)
            df_article = df_article.values
        else:
            df_article = encodings_dict[paper_id]            
 
        normalize = lambda x: x/np.linalg.norm(x, axis = 1, keepdims = True)
        query_emb = normalize(query_emb_init)
        val_norm = normalize(df_article)
        score_arr = safe_sparse_dot(val_norm, query_emb.T)

        # sort in decreasing order
        similar_paragr = sorted(zip(range(len(score_arr)), score_arr), 
                                key = lambda x: x[1], reverse=True)[:min(topk, len(score_arr))]      

        # obtain the text of this closest paragraphs
        field_text = text.split("\n")
        similar_paragr = [(field_text[idx], score) for (idx, score) in similar_paragr]

        return similar_paragr

