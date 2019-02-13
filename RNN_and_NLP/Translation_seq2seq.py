from __future__ import print_function, division
from builtins import range, input
import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
  Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K


import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU

  
# make sure we do softmax over the time axis
# expected shape is N x T x D
# note: the latest version of Keras allows you to pass in axis arg
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

# Basic configurations
BATCH_SIZE = 64  
EPOCHS = 100  
LATENT_DIMENSIONALITY = 256  
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_OF_WORDS = 20000
EMBEDDING_DIM = 100

# Separate data into inputs and targets
input_texts = [] 
target_texts = [] 
target_texts_inputs = []

NUM_TRAIN_SAMPLES = 10000  
num_samples = 0
# Data at: http://www.manythings.org/anki/
#file_path = '/content/drive/My Drive/spa (1).txt'
file_path_language = 'spa.txt'
for line in open(file_path_language, encoding="utf-8"):
  
  # keep track of limit of samples
  num_samples += 1
  if num_samples > NUM_TRAIN_SAMPLES:
    break

  # input and target are separated by tab
  if '\t' not in line:
    continue

  # Separate input text and the translation into our sec language
  input_text, translation = line.rstrip().split('\t')

  # Make target input and output using the sentence tags
  target_input = '<sos> ' + translation
  target = translation + ' <eos>'


  input_texts.append(input_text)
  target_texts.append(target)
  target_texts_inputs.append(target_input)

print("Number of samples:", len(input_texts))







# Tokenize inputs
tokenize_inputs = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenize_inputs.fit_on_texts(input_texts)
input_sequences = tokenize_inputs.texts_to_sequences(input_texts)

# Word-to-index mapping for our input language
word2idx_inputs = tokenize_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))


# Tokenize outputs, but don't filter out <sos> and <eos> 
tokenize_outputs = Tokenizer(num_words=MAX_NUM_OF_WORDS, filters='')
tokenize_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenize_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenize_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenize_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)


# Caculate max length input sequence for later padding 
max_len_input = max(len(s) for s in input_sequences)

# Padding sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')



# Create word-to-vec dict 
print('Loading word vectors...')
word2vec = {}
#path_word2vec = #'/content/drive/My Drive/glove.6B.%sd.txt'
path_word2vec = "glove.6B.%sd.txt"
with open(os.path.join(path_word2vec % EMBEDDING_DIM), encoding="utf-8") as f:
  # Space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))




# Embedding matrix
# Words not found word2vec become vectors of 0s
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_OF_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_OF_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector


# Vector/Embedding layer
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_input,
  # trainable=True
)


# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_categorical_encoded = np.zeros(
  (
    len(input_texts),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
    decoder_targets_categorical_encoded[i, t, word] = 1


##### build the model #####
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(
  LATENT_DIMENSIONALITY,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
encoder_outputs, h, c = encoder(x)
# encoder_outputs, h = encoder(x) #gru

# keep only the states to pass into decoder
encoder_states = [h, c]
# encoder_states = [state_h] # gru

# Set up the decoder, using [h, c] as initial state.
decoder_inputs_placeholder = Input(shape=(max_len_target,))

# this word embedding will not use pre-trained vectors
# although you could
decoder_embedding = Embedding(num_words_output, LATENT_DIMENSIONALITY)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have
# return_sequences=True
decoder_lstm = LSTM(
  LATENT_DIMENSIONALITY,
  return_sequences=True,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
decoder_outputs, _, _ = decoder_lstm(
  decoder_inputs_x,
  initial_state=encoder_states
)

# decoder_outputs, _ = decoder_gru(
#   decoder_inputs_x,
#   initial_state=encoder_states
# )

# final dense layer for predictions
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Create the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

# Compile the model and train it
model.compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r = model.fit(
  [encoder_inputs, decoder_inputs], decoder_targets_categorical_encoded,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2,
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# Save model
model.save('s2s.h5')

def decode_sequence(input_sequence):
  # Encode the input as state vectors.
  state_val = encoder_model.predict(input_sequence)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))

  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']

  # Create the translation
  output_sentence = []
  for _ in range(max_len_target):
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + state_val
    )
    # output_tokens, h = decoder_model.predict(
    #     [target_seq] + state_val
    # ) # gru

    # Get next word
    idx = np.argmax(output_tokens[0, 0, :])

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

    # Update states
    state_val = [h, c]
    # state_val = [h] # gru

  return ' '.join(output_sentence)



while True:
  # Do some test translations
  i = np.random.choice(len(input_texts))
  input_sequence = encoder_inputs[i:i+1]
  translation = decode_sequence(input_sequence)
  print('-')
  print('Input:', input_texts[i])
  print('Translation:', translation)

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break
