import os
import sys
import argparse
import collections
import json
import re

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None

embedding_dim = 300

def build_embeddings(embedding_file, vocab_file):
  model = json.load(open(embedding_file))
  vocabulary = [word.strip() for word in open(vocab_file)]
  embedding = np.zeros((len(vocabulary) + 1, embedding_dim))
  dictionary = {'<PAD>': 0}
  reverse_dictionary = {0: '<PAD>'}
  for i, word in enumerate(vocabulary[1:]):
    vec = np.asarray(model[word]).reshape(1, embedding_dim)
    embedding[i+1] = vec
    reverse_dictionary[i+1] = word
    dictionary[word] = i+1
  return embedding, dictionary, reverse_dictionary


def serialize_examples(question_file, dictionary):
  print('Generating tfrecord file from %s' %question_file)
  print('This may take up to half an hour')
  data = json.load(open(question_file))
  questions = [re.split(r'\s+', re.sub(r'([;?])', r'', q['question'])) for q in data['questions']]
  answers = [q['answer'].lower() for q in data['questions']]

  examples = [[dictionary[word.lower()] for word in q] for q in questions]
  labels = []
  answer_dict = {}
  for a in answers:
    if a not in answer_dict:
      answer_dict[a] = len(answer_dict)
    labels.append(answer_dict[a])

  reverse_answer_dict = dict(zip(answer_dict.values(), answer_dict.keys()))
  with open(os.path.join(FLAGS.datadir, 'answers.json'), 'w') as fp:
    json.dump(reverse_answer_dict, fp)

  assert len(labels) == len(examples), "Num examples does not match num labels"

  def make_example(question, answer):
    ex = tf.train.SequenceExample()
    ex.context.feature['length'].int64_list.value.append(len(question))
    ex.context.feature['answer'].int64_list.value.append(answer_dict[answer])
    fl_tokens = ex.feature_lists.feature_list['tokens']
    for token in question:
      fl_tokens.feature.add().int64_list.value.append(dictionary[token.lower()])
    return ex

  def update_progress(current, total):
    bar_length = 50
    text = ''
    progress = float(current)/total
    num_dots = int(round(bar_length*progress))
    num_spaces = bar_length - num_dots
    if current == total:
      text = '\r\nDone\r\n'
    else:
      text = '\r[{}] {:.2f}%  Serializing example {} of {}'.format('.'*num_dots + ' '*num_spaces, progress*100, current, total)
    sys.stdout.write(text)
    sys.stdout.flush()

  writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.datadir,'train_examples.tfrecords'))
  num_questions = len(questions)
  for i, (question, answer) in enumerate(zip(questions, answers)):
    ex = make_example(question, answer)
    writer.write(ex.SerializeToString())
    if i%100 == 0:
      update_progress(i+1, num_questions)
  update_progress(num_questions, num_questions)
  writer.close()


def read_and_decode(filename_queue):
  # with tf.name_scope('Input_Handlers'):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  context_features = {
    'length': tf.FixedLenFeature([], dtype=tf.int64), 
    'answer': tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example, context_features, sequence_features)
  return context_parsed['length'], sequence_parsed['tokens'], context_parsed['answer']


def generate_batch(batch_size, num_epochs=1):
  with tf.name_scope('Input_Handlers'):
    filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.datadir, 'train_examples.tfrecords')], num_epochs=num_epochs)
    length, question, answer = read_and_decode(filename_queue)

  with tf.name_scope('Batch_Generator'):
    lengths, questions, answers = tf.train.batch([length, question, answer], batch_size, dynamic_pad=True)

  return lengths, questions, answers


def print_batch(reverse_dictionary, answer_dict, questions, answers, predictions):
  for i in range(10):
    print('--------------------------------------------------------------------------------')
    for w_id in questions[i]:
      if w_id != 0:
        print(reverse_dictionary[w_id], end=' ')
      else:
        break
    print('')
    print('Predicted answer: %s Correct answer: %s' %(answer_dict[predictions[i]], answer_dict[answers[i]]))

def build_and_run_graph(embedding, dictionary, reverse_dictionary, answer_dict):

  # embedding, dictionary, reverse_dictionary = build_embeddings('embeddings.json', 'vocab.txt')
  print(dictionary)
  print(reverse_dictionary)
  print(answer_dict)

  batch_size = 64
  num_hidden = 512
  num_answers = 28
  num_epochs = 5000
  vocab_size = embedding.shape[0]


  lengths, questions, answers = generate_batch(batch_size, num_epochs)

  config = projector.ProjectorConfig()

  with tf.name_scope('Embedding_Ops'):
    embeddings = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=True, name="Embeddings")
    embeddings_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embeddings_init = embeddings.assign(embeddings_placeholder)

    embed = tf.nn.embedding_lookup(embeddings, questions)

    embedding_visual = config.embeddings.add()
    embedding_visual.tensor_name = embeddings.name

  with tf.name_scope('LSTM'):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden)
    output, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32, sequence_length=lengths)

  with tf.variable_scope('Fully_Connected_Output'):
      weights = tf.get_variable('weights', [num_hidden, num_answers], 
                  initializer=tf.random_normal_initializer(0.0, 1.0))
      biases = tf.get_variable('biases', [num_answers], initializer=tf.constant_initializer(0.0))
      logits = tf.matmul(final_state[0], weights) + biases
      tf.summary.histogram('weights', weights)
      tf.summary.histogram('biases', biases)

  with tf.name_scope('Evaluation'):
    prediction = tf.argmax(tf.nn.softmax(logits, name='Prediction'), 1)
    correct = tf.equal(prediction, answers)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='Accuracy')
    tf.summary.scalar('accuracy', accuracy)

  with tf.name_scope('Cross_Entropy'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answers))
    tf.summary.scalar('xent', loss)

  with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

  with tf.name_scope('Initialization'):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(embeddings_init, feed_dict={embeddings_placeholder: embedding})
    for i in range(num_epochs):
      sess.run(train_op)
      if i%50 == 0:
        s = sess.run(merged_summary)
        summary_writer.add_summary(s, i)
      if i%500 == 0:
        q, p, ans, acc = sess.run([questions, prediction, answers, accuracy])
        print('================================================================================')
        print('Epoch %d' %(i))
        print('================================================================================')
        print('Accuracy: %f' %(acc))
        print_batch(reverse_dictionary, answer_dict, q, ans, p)
        saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'), i)

    coord.request_stop()
    coord.join(threads)

def main(_):
  if not tf.gfile.Exists(FLAGS.datadir):
    sys.exit('No data directory found')
  
  if not tf.gfile.Exists(FLAGS.embedding_file):
    sys.exit('No embedding file found')
  elif not tf.gfile.Exists(FLAGS.vocab_file):
    sys.exit('No vocab file found')
  else:
    embedding, dictionary, reverse_dictionary = build_embeddings(FLAGS.embedding_file, FLAGS.vocab_file)

  if FLAGS.question_file == '':
    if not tf.gfile.Exists(os.path.join(FLAGS.datadir, 'train_examples.tfrecords')):
      sys.exit('No tfrecords file found. Use [--question_file] to generate a tfrecord')
  else:
    if tf.gfile.Exists(os.path.join(FLAGS.datadir, 'train_examples.tfrecords')):
      prompt = input('A tfrecords file already exists. Are you sure you want to generate a new one? The process may take up to half an hour. [y/n]: ')
      if prompt == 'y':
        tf.gfile.Rename(os.path.join(FLAGS.datadir, 'train_examples.tfrecords'), os.path.join(FLAGS.datadir, 'old_train_examples.tfrecords'))
        serialize_examples(FLAGS.question_file, dictionary)
    else:
      serialize_examples(FLAGS.question_file, dictionary)

  if tf.gfile.Exists(os.path.join(FLAGS.datadir, 'answers.json')):
    with open(os.path.join(FLAGS.datadir, 'answers.json')) as fp:
      data = json.load(fp)
      answer_dict = dict(zip([int(w_id) for w_id in data.keys()], data.values()))
  else:
    print('No answer dictionary found')

  if FLAGS.clear_logdir:
    if tf.gfile.Exists(FLAGS.logdir):
      tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, '1'))
    FLAGS.logdir = os.path.join(FLAGS.logdir, '1')
  else:
    if tf.gfile.Exists(FLAGS.logdir):
      runs = [int(folder) for folder in tf.gfile.ListDirectory(FLAGS.logdir)]
      new_run = max(runs) + 1
    else:
      new_run = 1
    tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, str(new_run)))
    FLAGS.logdir = os.path.join(FLAGS.logdir, str(new_run))
  
  build_and_run_graph(embedding, dictionary, reverse_dictionary, answer_dict)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', action='store_true', dest='clear_logdir',
                      help='Clears all runs in logdir')
  parser.add_argument('--logdir', type=str, default='logs/', 
                      help='Summaries log directory')
  parser.add_argument('--datadir', type=str, default='data/', 
                      help='Data directory')
  parser.add_argument('--question_file', type=str, default='',
                      help='JSON file with questions')
  parser.add_argument('--embedding_file', type=str, default='data/embeddings.json', 
                      help='JSON file with pretrained embedding dictionary')
  parser.add_argument('--vocab_file', type=str, default='data/vocab.tsv', 
                      help='.tsv file with vocabulary')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
