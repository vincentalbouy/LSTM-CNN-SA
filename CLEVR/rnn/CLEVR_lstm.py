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
  embedding = np.zeros((len(vocabulary), embedding_dim))
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
    fl_tokens = ex.feature_lists.feature_list['words']
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
  writer.close()


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


class RNN_Model:
  def __init__(self, config, graph=None):
    self.params = config
    if graph is not None:
      self.graph = graph
    else:
      self.graph = tf.Graph()
    self.saver = None
    self.projector = projector.ProjectorConfig()
    self.coord = None
    self.threads = None
    self.step = 0
    self.q_lengths = None
    self.questions = None
    self.init_op = None
    self.answers = None
    self.embedding_placeholder = None
    self.embedding_init_op = None
    self.embedding_lookup = None
    self.output = None
    self.final_state = None
    self.logits = None
    self.prediction = None
    self.correct = None
    self.accuracy = None
    self.loss = None
    self.train_op = None
    self.summary_writer = None
    self.summary_op = None

  def build_input_pipeline(self, input_files):
    with self.graph.name_scope('Input_Handlers'):
      filename_queue = tf.train.string_input_producer(input_files)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      context_features = {
        'length': tf.FixedLenFeature([], dtype=tf.int64),
        'answer': tf.FixedLenFeature([], dtype=tf.int64)
      }
      sequence_features = {
        'words': tf.FixedLenSequenceFeature([], dtype=tf.int64)
      }
      context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example, context_features, sequence_features)
      length = context_parsed['length']
      question = sequence_parsed['words']
      answer = context_parsed['answer']
    with self.graph.name_scope('Batch_Generator'):
      with self.graph.control_dependencies([length, question, answer]):
        self.q_lengths, self.questions, self.answers = tf.train.batch([length, question, answer], self.params['batch_size'], dynamic_pad=True)

  def build_embedding_layer(self, embedding):
    self.vocab_size = embedding.shape[0]
    with self.graph.name_scope('Embedding_Layer'):
      embeddings = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, embedding.shape[1]]), name='Embeddings')
      self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, embedding.shape[1]])
      self.embedding_init_op = embeddings.assign(self.embedding_placeholder)
      self.embedding_lookup = tf.nn.embedding_lookup(embeddings, self.questions)

      embedding_visual = self.projector.embeddings.add()
      embedding_visual.tensor_name = embeddings.name

  def build_rnn_layer(self):
    with self.graph.name_scope('LSTM'):
      cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['num_hidden'])
      self.output, self.final_state = tf.nn.dynamic_rnn(cell, self.embedding_lookup, dtype=tf.float32, sequence_length=self.q_lengths)

  def build_classifier(self):
    with tf.variable_scope('Classifier'):
      weights = tf.get_variable('weights', [self.params['num_hidden'], self.params['num_answers']],
                        initializer=tf.random_normal_initializer(0.0, 1.0))
      biases = tf.get_variable('biases', [self.params['num_answers']], initializer=tf.constant_initializer(0.0))
      self.logits = tf.matmul(self.final_state[0], weights) + biases
      tf.summary.histogram('weights', weights)
      tf.summary.histogram('biases', biases)

  def build_evaluation_layer(self):
    with self.graph.name_scope('Evaluation'):
      self.prediction = tf.argmax(tf.nn.softmax(self.logits, name='Prediction'), 1)
      self.correct = tf.equal(self.prediction, self.answers)
      self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='Accuracy')
      tf.summary.scalar('accuracy', self.accuracy)
    with self.graph.name_scope('Cross_Entropy'):
      self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.answers))
      tf.summary.scalar('xent', self.loss)

  def build_training_op(self):
    with self.graph.name_scope('Training'):
      self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

  def build_init_op(self):
    with tf.name_scope('Initialization'):
      self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  def build_summary_op(self, logdir):
    self.summary_writer = tf.summary.FileWriter(logdir, graph=self.graph)
    self.summary_op = tf.summary.merge_all()

  def build_graph(self, embedding, logdir, input_files):
    with self.graph.as_default():
      self.build_input_pipeline(input_files)
      self.build_embedding_layer(embedding)
      self.build_rnn_layer()
      self.build_classifier()
      self.build_evaluation_layer()
      self.build_training_op()
      self.build_init_op()
      self.build_summary_op(logdir)

  def initialize_graph(self, session, embedding):
    with self.graph.as_default():
      session.run(self.init_op)
      session.run(self.embedding_init_op, feed_dict={self.embedding_placeholder: embedding})
      self.saver = tf.train.Saver()
      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(sess=session, coord=self.coord)
      projector.visualize_embeddings(self.summary_writer, self.projector)

  def train_step(self, session, reverse_dictionary, answer_dict, logdir):
    with self.graph.as_default():
      if self.step%10 == 0:
        summary = session.run(self.summary_op)
        self.summary_writer.add_summary(summary, self.step)
      if self.step%100 == 0:
        q, p, ans, acc = session.run([self.questions, self.prediction, self.answers, self.accuracy])
        print('================================================================================')
        print('Epoch %d' %(self.step))
        print('================================================================================')
        print('Accuracy: %f' %(acc))
        print_batch(reverse_dictionary, answer_dict, q, ans, p)
        self.saver.save(session, os.path.join(logdir, 'model.ckpt'), self.step)
      session.run(self.train_op)
      self.step = self.step + 1

  def run_training(self, session, reverse_dictionary, answer_dict, logdir):
    for i in range(self.params['num_epochs'] + 1):
      self.train_step(session, reverse_dictionary, answer_dict, logdir)
    self.coord.request_stop()
    self.coord.join(self.threads)
  
  def get_graph(self):
    return self.graph


def build_and_run_graph(config, embedding, logdir, input_files, reverse_dictionary, answer_dict):
  model = RNN_Model(config)
  model.build_graph(embedding, logdir, input_files)
  with tf.Session(graph=model.get_graph()) as sess:
    model.initialize_graph(sess, embedding)
    model.run_training(sess, reverse_dictionary, answer_dict, logdir)


def main(_):
  config = {
      'batch_size': 64,
      'num_hidden': 512,
      'num_epochs': 5000,
      'num_answers': 28
  }
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
  elif not tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, '1'))
    FLAGS.logdir = os.path.join(FLAGS.logdir, '1')
  else:
    runs = [int(folder) for folder in tf.gfile.ListDirectory(FLAGS.logdir)]
    new_run = max(runs) + 1
    tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, str(new_run)))
    FLAGS.logdir = os.path.join(FLAGS.logdir, str(new_run))

  if tf.gfile.Exists(FLAGS.config_file) and FLAGS.config_file != '':
    with open(FLAGS.config_file) as fp:
      config = json.load(fp)
  else:
    print('No config file found, using default configuration')
  
  build_and_run_graph(config, embedding, FLAGS.logdir, [os.path.join(FLAGS.datadir, 'train_examples.tfrecords')], reverse_dictionary, answer_dict)


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
  parser.add_argument('--config_file', type=str, default='', 
                      help='JSON file with model configuration')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
