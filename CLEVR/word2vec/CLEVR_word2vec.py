# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Larry Chen
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
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import random
import re
import json
import argparse

from collections import OrderedDict

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None
data_index = 0


def read_data(filename):
  with open(filename) as data_file:    
    data = json.load(data_file)
    questions = data['questions']
  return questions


def build_dataset(questions):
  dictionary = {}
  question_vocab = {}
  answer_vocab = {}
  data = []

  for q in questions:
    question_words = re.split(r'\s+|[,;?.-]\s*', q['question'])
    answer_word = q['answer'].lower()

    for word in question_words:
      w = word.lower()
      if len(w) is not 0:
        if w in question_vocab:
          question_vocab[w] += 1
        else:
          question_vocab[w] = 1
        if w not in dictionary:
          dictionary[w] = len(dictionary)
        index = dictionary[w]
        data.append(index)

    if answer_word in answer_vocab:
      answer_vocab[answer_word] += 1
    else:
      answer_vocab[answer_word] = 1
    if answer_word not in dictionary:
      dictionary[answer_word] = len(dictionary)
    index = dictionary[answer_word]

    data.append(index)

  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, dictionary, reversed_dictionary, question_vocab, answer_vocab


def train(data, dictionary, reverse_dictionary):
  vocabulary_size = len(dictionary)

  # Function to generate a training batch for the skip-gram model.
  def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

  
  # Build a skip-gram model

  batch_size = 64
  embedding_size = 64  # Dimension of the embedding vector.
  skip_window = 1       # How many words to consider left and right.
  num_skips = 2         # How many times to reuse an input to generate a label.

  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 93  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  num_sampled = 64    # Number of negative examples to sample.

  graph = tf.Graph()
  config = projector.ProjectorConfig()

  with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Add embeddings to tensorboard
      embedding = config.embeddings.add()
      embedding.tensor_name = embeddings.name
      # embedding.metadata_path = os.path.join('logs', 'metadata.tsv')

      # Construct the variables for the NCE loss
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

  # Begin training
  
  num_steps = 100001

  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    summary_writer = tf.summary.FileWriter('logs', session.graph)
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver()

    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(
          batch_size, num_skips, skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0
        saver.save(session, os.path.join('logs', "model.ckpt"), step)

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  questions = read_data(FLAGS.data_file)

  print('Number of questions: %d' %(len(questions)))

  data, dictionary, reverse_dictionary, question_vocab, answer_vocab = build_dataset(questions)

  vocabulary_size = len(dictionary)

  sorted_q_vocab = OrderedDict(sorted(question_vocab.items(), key=lambda t: t[1], reverse=True))
  sorted_a_vocab = OrderedDict(sorted(answer_vocab.items(), key=lambda t: t[1], reverse=True))

  # Write labels to metadata.tsv
  file = open(os.path.join(FLAGS.log_dir, 'metadata.tsv'), 'w')
  for label in reverse_dictionary.values():
    file.write("%s\n" % label)
  file.close()

  print('Most common question words:', list(sorted_q_vocab.items())[:10])
  print('Most common answers:', list(sorted_a_vocab.items())[:10])
  print('Sample data:', data[:10], [reverse_dictionary[i] for i in data[:10]])
  print('Vocabulary size: ', vocabulary_size)

  train(data, dictionary, reverse_dictionary)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_file', type=str, required=True, 
                      help='The file with question data')
  parser.add_argument('--log_dir', type=str, default='logs/',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
