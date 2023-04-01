#Importing libraries
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Global parameters
root_folder='.'
DATA_PATH = '../data'

# Both train and test set are in the root data directory
train_path = DATA_PATH
test_path = DATA_PATH

# Parameters for our model
ATTENTION_FUNC='general'
# TO DO: Fill in more model details. Maybe add command line settings

########################## CLASSES ###################################

class Generator(tf.keras.Model):
  def __init__(self,hidden_dim):
    super().__init__()
    self.attention = LuongAttention(hidden_dim, attention_func)
    self.hidden_dim = hidden_dim
    self.lstm=tf.keras.layers.LSTM(hidden_dim,return_sequences=True,return_state=True) #TO DO: Change to bidirectional
    # TO DO: Add convolutional layer

  def call(self,input_sequence,states):
    #LSTM
    output, state_h, state_c = self.lstm(input_sequence, initial_state=states) #h-memory c-carry
    return output, state_h, state_c

  def init_states(self, batch_size):
    # Return a all 0s initial states
    return (tf.zeros([batch_size, self.hidden_dim]),tf.zeros([batch_size, self.hidden_dim]))

class Discriminator(tf.keras.Model):
  def __init__(self,vocab_size, embedding_dim, hidden_dim, attention_func):
    super(Discriminator,self).__init__()
    self.attention = LuongAttention(hidden_dim, attention_func)
    self.hidden_dim = hidden_dim
    self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
    self.wc = tf.keras.layers.Dense(hidden_dim, activation='tanh')
    self.ws = tf.keras.layers.Dense(vocab_size)

  def call(self,input_sequence_word,state_for_this_unit,generator_outputs):
      # Remember that the input to the discriminator
      # is now a batch of one-word sequences,
      # which means that its shape is (batch_size, 1)
      embed = self.embedding(input_sequence_word)

      # Therefore, the lstm_out has shape (batch_size, 1, hidden_dim)
      lstm_out, state_h, state_c = self.lstm(embed, initial_state=state_for_this_unit)

      # Use self.attention to compute the context and alignment vectors
      # context vector's shape: (batch_size, 1, hidden_dim)
      # alignment vector's shape: (batch_size, 1, source_length)
      context, alignment = self.attention(lstm_out, generator_outputs)

      # Combine the context vector and the LSTM output
      # Before combined, both have shape of (batch_size, 1, hidden_dim),
      # so let's squeeze the axis 1 first
      # After combined, it will have shape of (batch_size, 2 * hidden_dim)

      lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], axis=1)

      # lstm_out now has shape (batch_size, hidden_dim), because we passed it through a dense layer of size (hidden_dim)
      lstm_out = self.wc(lstm_out)

      # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
      logits = self.ws(lstm_out)

      return logits, state_h, state_c, alignment

#Will use this in the new Discriminator class

class LuongAttention(tf.keras.Model):
    def __init__(self, hidden_dim, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Attention score must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(hidden_dim) 
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(hidden_dim, activation='tanh')
            self.va = tf.keras.layers.Dense(1)
        # TO DO: add dot attention function? 


    def call(self,discriminator_output,generator_outputs):
      #We use the discriminator output(hidden state) from one unit at a time & use it along with ALL generator outputs 
      # Since we use all generator outputs we are using Global Attention
      if self.attention_func == 'dot':
            # Dot score function: discriminator_output (dot) generator_output
            # discriminator_output has shape: (batch_size, 1, hidden_dim)
            # generator_outputs has shape: (batch_size, max_len, hidden_dim)
            score = tf.matmul(discriminator_output, generator_outputs, transpose_b=True) # (batch_size, 1, max_len)
            #So what does this mean? The score for each unit that we have in the generator i.e generator quality 
      elif self.attention_func == 'general':
          # General score function: discriminator_output (dot) (Wa (dot) generator_output)
          # Dense layer doesn't change dims of generator_output since we use Dense(hidden_dim)
          # discriminator_output has shape: (batch_size, 1, hidden_dim)
          # generator_output has shape: (batch_size, max_len, hidden_dim)
          score = tf.matmul(discriminator_output, self.wa(generator_outputs), transpose_b=True) #(batch_size, 1, max_len)
      elif self.attention_func == 'concat':
          # Concat score function: va (dot) tanh(Wa (dot) concat(discriminator_output + generator_output))
          # Discriminator output must be broadcasted to generator output's shape first
          # tf.tile's second argument [1, generator_output.shape[1], 1] basically multiplies 
          # discriminator_output's current dimensions position wise by the numbers in the second argument
          discriminator_output = tf.tile(discriminator_output, [1, generator_outputs.shape[1], 1]) #shape (batch size, max len,hidden_dim)

          # Concat => Wa => va
          # (batch_size, max_len, 2 * hidden_dim) => (batch_size, max_len, hidden_dim) => (batch_size, max_len, 1)

          score = self.va(self.wa(tf.concat((discriminator_output, generator_outputs), axis=-1))) # (batch_size, max len, 1)

          # Transpose score vector to have the same shape as other two above
          # (batch_size, max_len, 1) => (batch_size, 1, max_len)
          score = tf.transpose(score, [0, 2, 1]) #(batch_size, 1, max_len)

        # alignment a_t = softmax(score), focuses where to pay attention in the sentence 
      alignment = tf.keras.activations.softmax(score, axis=-1) #(batch_size, 1, max_len)
      
      # context vector c_t is the weighted average sum of generator outputs, score the hidden states
      # since each generator hidden states is most associated with a certain word in the input sentence
      #(batch_size, 1, max_len) X (batch_size, max_len , hidden_dim)
      context = tf.matmul(alignment, generator_outputs) # (batch_size, 1, hidden_dim)

      return context, alignment
  
  


#######################    OTHER FUNCTIONS

def make_dataset():
    # Define a dataset 
    dataset = tf.data.Dataset.from_tensor_slices((generator_inputs, discriminator_inputs, discriminator_targets)) # we might not want to do this? 
    for element in dataset:
        print(element)
        print(len(element))
        print(type(element[0]))
        print(type(element[1]))
        print(type(element[2]))
        break

    #dataset = dataset.shuffle(len(input_data)).batch(BATCH_SIZE, drop_remainder=True)  I think our dataset doesn't work for this
    #Drop remainder=True drops batches with size < batch_size




generator=Generator(num_words_inputs,EMBEDDING_DIM,HIDDEN_DIM)
initial_states = generator.init_states(1) # 1 X 1024 (HIDDEN_DIM). All zeroes
test_generator_output = generator(tf.constant([[1, 23, 4, 5, 0, 0]]), initial_states) 


discriminator=Discriminator(num_words_outputs,EMBEDDING_DIM,HIDDEN_DIM)
de_initial_state = test_generator_output[1:] # H-memory & C-cell
test_discriminator_output= discriminator(tf.constant([[1, 3, 5, 7, 9, 0, 0, 0]]), de_initial_state)
print(test_discriminator_output)
#  Outputs  Note 1 is batch size here 

# 0 For each unit, for each word in vocab (7169 size) a probability thus batch_size X len_seq X vocab_size (output)
# 1 H-memory 1 X HIDDEN_DIM
# 2 C-memory 1 X HIDDEN_DIM

def loss_func(targets,logits):
  #SparseCategoicalCrossentropy doesn't require targets to be one hot encoded & is more efficient than categorical cross entropy when num_classes is huge
  crossentropy=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #If from_logits=True no need to apply softmax in dense layer
  #Masking
  mask = tf.math.logical_not(tf.math.equal(targets, 0))
  mask = tf.cast(mask, dtype=tf.int64) #Convert boolean to int
  loss = crossentropy(targets, logits, sample_weight=mask)
  return loss

def accuracy_fn(y_true, y_pred):
    # y_pred shape is batch_size, max length, vocab size
    # y_true shape is batch_size, max length
    pred_values = K.cast(K.argmax(y_pred, axis=-1), dtype='int32') #returns batch_size,max length
    correct = K.cast(K.equal(y_true, pred_values), dtype='float32')

    # 0 is padding, don't include those
    mask = K.cast(K.greater(y_true, 0), dtype='float32') #Shape batch_size,max len 
    n_correct = K.sum(mask * correct)  #Sums across all axes if not specified, thus shape is ()
    n_total = K.sum(mask) #Real length of sequences in entire batch
  
    return n_correct / n_total



# Create the main train function
def main_train(generator, discriminator, dataset, n_epochs, batch_size, optimizer, checkpoint, checkpoint_prefix):
    '''
    dataset is of type tensorflow.python.data.ops.dataset_ops.BatchDataset 
    '''
    losses = []
    accuracies = []

    for e in range(n_epochs):
        # Get the initial time
        start = time.time()
        # Get the initial state for the generator
        en_initial_states = generator.init_states(batch_size)
        # For every batch data
        for batch, (input_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            # Train and get the loss value 
            loss, accuracy = train_step(input_seq, target_seq_in, target_seq_out, en_initial_states, optimizer)
        
            if batch % 100 == 0:
                # Store the loss and accuracy values
                losses.append(loss)
                accuracies.append(accuracy)
                print('Epoch {} Batch {} Loss {:.4f} Acc:{:.4f}'.format(e + 1, batch, loss.numpy(), accuracy.numpy()))
                
        # saving (checkpoint) the model every 2 epochs
        if (e + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    
        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))
        
    return losses, accuracies

# Create an Adam optimizer and clips gradients by norm
optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
# Create a checkpoint object to save the model
checkpoint_dir = './training_ckpt_GAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


#Will use this in the new Discriminator class

class LuongAttention(tf.keras.Model):
    def __init__(self, hidden_dim, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Attention score must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(hidden_dim) 
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(hidden_dim, activation='tanh')
            self.va = tf.keras.layers.Dense(1)
        # TO DO: add dot attention function? 


    def call(self,discriminator_output,generator_outputs):
      #We use the discriminator output(hidden state) from one unit at a time & use it along with ALL generator outputs 
      # Since we use all generator outputs we are using Global Attention
      if self.attention_func == 'dot':
            # Dot score function: discriminator_output (dot) generator_output
            # discriminator_output has shape: (batch_size, 1, hidden_dim)
            # generator_outputs has shape: (batch_size, max_len, hidden_dim)
            score = tf.matmul(discriminator_output, generator_outputs, transpose_b=True) # (batch_size, 1, max_len)
            #So what does this mean? The score for each unit that we have in the generator i.e generator quality 
      elif self.attention_func == 'general':
          # General score function: discriminator_output (dot) (Wa (dot) generator_output)
          # Dense layer doesn't change dims of generator_output since we use Dense(hidden_dim)
          # discriminator_output has shape: (batch_size, 1, hidden_dim)
          # generator_output has shape: (batch_size, max_len, hidden_dim)
          score = tf.matmul(discriminator_output, self.wa(generator_outputs), transpose_b=True) #(batch_size, 1, max_len)
      elif self.attention_func == 'concat':
          # Concat score function: va (dot) tanh(Wa (dot) concat(discriminator_output + generator_output))
          # Discriminator output must be broadcasted to generator output's shape first
          # tf.tile's second argument [1, generator_output.shape[1], 1] basically multiplies 
          # discriminator_output's current dimensions position wise by the numbers in the second argument
          discriminator_output = tf.tile(discriminator_output, [1, generator_outputs.shape[1], 1]) #shape (batch size, max len,hidden_dim)

          # Concat => Wa => va
          # (batch_size, max_len, 2 * hidden_dim) => (batch_size, max_len, hidden_dim) => (batch_size, max_len, 1)

          score = self.va(self.wa(tf.concat((discriminator_output, generator_outputs), axis=-1))) # (batch_size, max len, 1)

          # Transpose score vector to have the same shape as other two above
          # (batch_size, max_len, 1) => (batch_size, 1, max_len)
          score = tf.transpose(score, [0, 2, 1]) #(batch_size, 1, max_len)

        # alignment a_t = softmax(score), focuses where to pay attention in the sentence 
      alignment = tf.keras.activations.softmax(score, axis=-1) #(batch_size, 1, max_len)
      
      # context vector c_t is the weighted average sum of generator outputs, score the hidden states
      # since each generator hidden states is most associated with a certain word in the input sentence
      #(batch_size, 1, max_len) X (batch_size, max_len , hidden_dim)
      context = tf.matmul(alignment, generator_outputs) # (batch_size, 1, hidden_dim)

      return context, alignment

@tf.function
def train_step(input_seq, target_seq_in, target_seq_out, en_initial_states, optimizer):
    ''' A training step, train a batch of the data and return the loss value reached
        Input:
        - input_seq: array of integers, shape [batch_size, max_seq_len].
            the input sequence
        - target_seq_out: array of integers, shape [batch_size, max_seq_len].
            the target seq, our target sequence
        - target_seq_in: array of integers, shape [batch_size, max_seq_len].
            the input sequence to the discriminator, we use Teacher Forcing
        - en_initial_states: tuple of 2 arrays of shape [batch_size, hidden_dim] each.
            the initial state of the generator for h & c state
        - optimizer: a tf.keras.optimizers.
        Output:
        - loss: loss value
        
    '''
    loss = 0.
    acc = 0.
    logits = None
    
    with tf.GradientTape() as tape:
        en_outputs = generator(input_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        #For attention for each word in the output we score all the generator outputs
        # to see which ones we should pay attention to for our current word
        # Loop through all our words in our target sequence to see which words in the 
        # input sequence matter
        # Note that the discriminator states (h & c) are getting updated after every unit predicts a logit
        for i in range(target_seq_out.shape[1]):
            # Input to the discriminator must have shape of (batch_size, 1)
            # so we need to expand one dimension
            # as target_seq_in[:, i] gives us a shape (batch_size,)
            discriminator_in = tf.expand_dims(target_seq_in[:, i], 1)

            #Passing all generator outputs since we use GLOBAL ATTENTION

            logit, de_state_h, de_state_c, _ = discriminator(discriminator_in, (de_state_h, de_state_c), en_outputs[0])

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)
            # Store the logits to calculate the accuracy
            # Adds a shape of 1 at position axis
            logit = K.expand_dims(logit, axis=1)

            #Now we need to concatenate logit max_len times to get logits for all the whole sequences in the batch
            #This concatenation needs to happen target_seq_out.shape[1] times
            if logits is None:
                logits = logit
            else:
                logits = K.concatenate((logits,logit), axis=1) 
        # After the loop logits' shape is batch_size,max_len,vocab_size

        # Calculate the accuracy for the batch data        
        acc = accuracy_fn(target_seq_out, logits)
    # Update the parameters and the optimizer
    variables = generator.trainable_variables + discriminator.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    #Normalise by max_len since we performed loss addition max_len times
    return loss / target_seq_out.shape[1], acc

#Create the generator
generator = Generator(num_words_inputs, EMBEDDING_DIM, HIDDEN_DIM)
discriminator = Discriminator(num_words_output, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_FUNC)

# Create an Adam optimizer and clips gradients by norm
optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
# Create a checkpoint object to save the model
checkpoint_dir = './training_ckpt_seq2seq_att'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

losses, accuracies = main_train(generator, discriminator, dataset, EPOCHS, BATCH_SIZE, optimizer, checkpoint, checkpoint_prefix)


def predict_seq2seq_att(input_text, input_max_len, tokenizer_inputs, word2idx_outputs, idx2word_outputs):
    if input_text is None:
        input_text = input_data[np.random.choice(len(input_data))]
    print(input_text)
    # Tokenize the input text
    input_seq = tokenizer_inputs.texts_to_sequences([input_text])
    # Pad the sentence
    input_seq = pad_sequences(input_seq, maxlen=input_max_len, padding='post')
    # Get the generator initial states
    en_initial_states = generator.init_states(1)
    # Get the generator outputs or hidden states
    en_outputs = generator(tf.constant(input_seq), en_initial_states)
    # Set the discriminator input to the sos token
    de_input = tf.constant([[word2idx_outputs['<sos>']]])
    # Set the initial hidden states of the discriminator to the hidden states of the generator
    de_state_h, de_state_c = en_outputs[1:]
    
    out_words = []
    alignments = []

    while True:
        # Get the discriminator with attention output
        de_output, de_state_h, de_state_c, alignment = discriminator(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        # Detokenize the output
        out_words.append(idx2word_outputs[de_input.numpy()[0][0]])
        # Save the aligment matrix
        alignments.append(alignment.numpy())

        if out_words[-1] == '<eos>' or len(out_words) >= 20:
            break
    # Join the output words
    print(' '.join(out_words))
    return np.array(alignments), input_text.split(' '), out_words
