from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
import tensorflow as tf

from keras.models import load_model, Model
import keras.backend as K

from keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from seqGAN_utils import softmax
import numpy as np

"""
    inspired by: https://github.com/mshadloo/Neural-Machine-Translation-with-Attention/blob/master/neuralMT.py
 """
 
def lstm(units,return_sequences=False, return_state=False):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    recurrent_initializer='glorot_uniform')
  else:
    return LSTM(units, return_sequences=return_sequences, return_state=return_state,  recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')

def bidirectional(units,return_sequences=False, return_state=False):
    return  Bidirectional(lstm(units,return_sequences, return_state))

rnn_archs = {'lstm': lstm, 'bidirectional':bidirectional}

class AttentionLayer:
    def __init__(self, Tx):
        self.repeator = RepeatVector(Tx)
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(1024, activation="tanh")
        self.densor2 = Dense(1)
        self.activator = Activation(softmax,name='attention_weights')  # Custom softmax from utils
        self.dotor = Dot(axes=1)

    def one_step_attention(self,a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
        s_prev = self.repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = self.concatenator([a, s_prev])
        e = self.densor1(concat)
        energies = self.densor2(e)
        alphas = self.activator(energies)
        context = self.dotor([alphas, a])

        return context

class GAN_Model:
    def stack(outputs):
        outputs = K.stack(outputs)
        return K.permute_dimensions(outputs, pattern=(1, 0, 2))

    def __init__(self, rnn_arch, Tx, Ty, num_hidden_units, input_size):
        """initializes the model

        Args:
            Tx (int I think?): Idk what this does. It might be a set length for generation?, if so we should delete. It is defined in utils as input length?
            Ty (int I think?): Idk what this does. It is defined in utils as target length?
            rnn_arch (string): the architecture we're using, ex: lstm, bidirectional
            num_hidden_units (int): how many hidden units in the generator
            input_size (int): input size
        """

        # import pdb; pdb.set_trace()
        self.rnn_arch = rnn_arch
        self.discriminator_units = num_hidden_units

        # attention
        self.attentionLayer = AttentionLayer(Tx)

        # generator
        self.input = Input(shape=(Tx,))
        generator = rnn_archs[rnn_arch](num_hidden_units, return_sequences=True)
        self.generator_out = generator(input_size)

        # discriminator
        self.discriminator = lstm(units=num_hidden_units, return_state=True)
        self.dense_decode = Dense(input_size, activation='softmax')

        # concat
        self.concat2 = Concatenate(axis=2)

        self.discriminator_state_0 = Input(shape=(num_hidden_units,))
        self.discriminator_cell_0 = Input(shape=(num_hidden_units,))
        self.train_model = self.get_train_model(Ty)
        print('train model was built')
        
        self.inference_model = self.get_inference_model(Ty)
        print('inference model was built')

    def get_train_model(self, Ty):
        discriminator_inp = Input(shape=(Ty,))
        discriminator_inp_embedded = self.discriminator_embedding(discriminator_inp)

        discriminator_state = self.discriminator_state_0
        discriminator_cell = self.discriminator_cell_0

        # Iterate attention Ty times
        outputs = []
        for t in range(Ty):

            # Get context vector with generator and attention
            context = self.attentionLayer.one_step_attention(self.generator_out, discriminator_state)

            # For teacher forcing, get the previous word
            select_layer = Lambda(lambda x: x[:, t:t + 1])
            prevWord = select_layer(discriminator_inp_embedded)

            # Concat context and previous word as discriminator input
            discriminator_in_concat = self.concat2([context, prevWord])

            # pass into discriminator, inference output
            pred, discriminator_state, discriminator_cell = self.discriminator(discriminator_in_concat, initial_state=[discriminator_state, discriminator_cell])
            pred = self.dense_decode(pred)
            outputs.append(pred)

        stack_layer = Lambda(GAN_Model.stack)
        outputs = stack_layer(outputs)
        return Model(inputs=[self.input, discriminator_inp, self.discriminator_state_0, self.discriminator_cell_0], outputs=outputs)

    # in the inference model teacher forcing is not available
    def get_inference_model(self, Tx):
        discriminator_inp = Input(shape=(1,))

        discriminator_state = self.discriminator_state_0
        discriminator_cell = self.discriminator_cell_0

        discriminator_inp_embedded = self.discriminator_embedding(discriminator_inp)
        # Get context vector with generator and attention
        context = self.attentionLayer.one_step_attention(self.generator_out, discriminator_state)

        # Concat context and previous word as discriminator input
        discriminator_in_concat = self.concat2([context, discriminator_inp_embedded])

        # pass into discriminator, inference output
        if self.rnn_arch == 'gru':
            pred, discriminator_state = self.discriminator(discriminator_in_concat, initial_state=discriminator_state)
        else:
            pred, discriminator_state, discriminator_cell = self.discriminator(discriminator_in_concat,
                                                             initial_state=[discriminator_state, discriminator_cell])

        pred = self.dense_decode(pred)

        if self.rnn_arch == 'gru':
            return Model(inputs=[self.input, discriminator_inp, self.discriminator_state_0], outputs=pred)
        else:
            return Model(inputs=[self.input, discriminator_inp, self.discriminator_state_0, self.discriminator_cell_0], outputs=pred)

    def fit(self, enc_inp, discriminator_inp, targ, batch_size=64, verbose=0):

        s_0 = np.zeros((len(enc_inp), self.discriminator_units))
        c_0 = np.zeros((len(enc_inp), self.discriminator_units))
        history = self.train_model.fit([enc_inp, discriminator_inp, s_0, c_0], targ, batch_size=batch_size,verbose=verbose)
        self.inference_model.set_weights(self.train_model.get_weights())
        return history

    def compile(self, opt, loss, metrics):
        self.train_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.inference_model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def evaluate(self, enc_inp, discriminator_inp, targ, batch_size=64, verbose=0):
        s_0 = np.zeros((len(enc_inp), self.discriminator_units))
        c_0 = np.zeros((len(enc_inp), self.discriminator_units))
        return self.train_model.evaluate([enc_inp, discriminator_inp, s_0, c_0], targ, batch_size=batch_size, verbose=verbose)

    def inference_evaluate(self, enc_inp, discriminator_inp, targ, batch_size=64, verbose=0):
        s_0 = np.zeros((len(enc_inp), self.discriminator_units))
        c_0 = np.zeros((len(enc_inp), self.discriminator_units))

        Ty = targ.shape[1]
        loss = 0.0
        acc = 0.0
        preds = []

        for t in range(Ty):
            pred = self.inference_model.predict([enc_inp, discriminator_inp, s_0, c_0])
            loss_b, acc_b = self.inference_model.evaluate([enc_inp, discriminator_inp, s_0, c_0], targ[:, t],
                                                            batch_size=batch_size, verbose=verbose)

            pred = np.argmax(pred, axis=-1)
            discriminator_inp = np.expand_dims(pred, axis=1)
            preds.append(discriminator_inp)
            loss += loss_b
            acc += acc_b
        return loss / Ty, acc / Ty, GAN_Model.stack(preds).numpy()

    def save(self, train_file, infer_file):
        self.train_model.save(train_file)
        self.inference_model.save(infer_file)
