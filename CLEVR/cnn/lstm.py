import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def lstm(_input, lengths, embeddings):
  with tf.variable_scope('Embedding'):
    init_embedding = tf.constant_initializer(embeddings)
    embeddings = tf.get_variable('Embeddings', [94, 300], initializer=init_embedding)
    out = tf.nn.embedding_lookup(embeddings, _input)

  with tf.variable_scope('LSTM'):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
    outputs, final_state = tf.nn.dynamic_rnn(cell, out, dtype=tf.float32, sequence_length=lengths)
    cell_state, hidden_state = final_state
    # print('OUTPUT')
    # print(output)
    # print('')
    # print('CELL_STATES')
    # print(cell_states)
    # print('')
    # print('HIDDEN_STATES')
    # print(hidden_state)
    # print('')
  return cell_state
