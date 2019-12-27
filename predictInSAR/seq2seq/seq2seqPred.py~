

import numpy as np
import keras


from utils import random_sine, plot_prediction




keras.backend.clear_session()

layers = [35, 35] # Number of hidden neuros in each layer of the encoder and decoder

learning_rate = 0.01
decay = 0 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

num_input_features = 1 # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1 # The dimensionality of the output at each time step. In this case a 1D signal.

loss = "mse" # Other loss functions are possible, see Keras documentation.

lambda_regulariser = 0.000001 # Will not be used if regulariser is None
regulariser = None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

batch_size = 512
steps_per_epoch = 200 # batch_size * steps_per_epoch = total number of training examples
epochs = 4

input_sequence_length = 15 # Length of the sequence used by the encoder
target_sequence_length = 15 # Length of the sequence predicted by the decoder
num_steps_to_predict = 20 # Length to use when testing the model

num_signals = 2 # The number of random sine waves the compose the signal. The more sine waves, the harder the problem.



# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

encoder_states = encoder_outputs_and_states[1:]



decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

decoder_outputs = decoder_outputs_and_states[0]

decoder_dense = keras.layers.Dense(num_output_features,
                                   activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)


model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)

train_data_generator = random_sine(batch_size=batch_size,
                                   steps_per_epoch=steps_per_epoch,
                                   input_sequence_length=input_sequence_length,
                                   target_sequence_length=target_sequence_length,
                                   min_frequency=0.1, max_frequency=10,
                                   min_amplitude=0.1, max_amplitude=1,
                                   min_offset=-0.5, max_offset=0.5,
                                   num_signals=num_signals, seed=1969)

model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)


test_data_generator = random_sine(batch_size=1000,
                                  steps_per_epoch=steps_per_epoch,
                                  input_sequence_length=input_sequence_length,
                                  target_sequence_length=target_sequence_length,
                                  min_frequency=0.1, max_frequency=10,
                                  min_amplitude=0.1, max_amplitude=1,
                                  min_offset=-0.5, max_offset=0.5,
                                  num_signals=num_signals, seed=2000)

(x_encoder_test, x_decoder_test), y_test = next(test_data_generator) # x_decoder_test is composed of zeros.

y_test_predicted = model.predict([x_encoder_test, x_decoder_test])


encoder_predict_model = keras.models.Model(encoder_inputs,
                                           encoder_states)

decoder_states_inputs = []

for hidden_neurons in layers[::-1]:
    # One state for GRU
    decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

decoder_outputs_and_states = decoder(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_predict_model = keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)


# In[10]:


# Let's define a small function that predicts based on the trained encoder and decoder models

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
    """Predict time series with encoder-decoder.

    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.

    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict

    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 1))


    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
        [decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)



# In[11]:


# The aim of this tutorial isn't to present how to evaluate the model or investigate the training.
# We could plot evaluation metrics such as RMSE over time, compare train and test batches for overfitting,
# produce validation and learning curves to analyse the effect of the number of epochs or training examples,
# have fun playing with tensorboard etc... We would need at least a whole other post for this.
# However, let's at least make sure that our model can predict correctly...
# Ask the generator to produce a batch of samples, don't forget to set the seed to something other than what was
# used for training or you will be testing on train data.
# The next function asks the generator to produce it's first batch.

test_data_generator = random_sine(batch_size=1000,
                                  steps_per_epoch=steps_per_epoch,
                                  input_sequence_length=input_sequence_length,
                                  target_sequence_length=target_sequence_length,
                                  min_frequency=0.1, max_frequency=10,
                                  min_amplitude=0.1, max_amplitude=1,
                                  min_offset=-0.5, max_offset=0.5,
                                  num_signals=num_signals, seed=2000)

(x_test, _), y_test = next(test_data_generator)


# In[12]:


y_test_predicted = predict(x_test, encoder_predict_model, decoder_predict_model, num_steps_to_predict)

# Select 10 random examples to plot
indices = np.random.choice(range(x_test.shape[0]), replace=False, size=10)


for index in indices:
    plot_prediction(x_test[index, :, :], y_test[index, :, :], y_test_predicted[index, :, :])

# The model seems to struggle on very low wave signals. But that makes sense, the model doesn't see enough of the signal
# to make a good estimation of the frequency components.


# In[14]:


train_data_generator = random_sine(batch_size=1000,
                                   steps_per_epoch=steps_per_epoch,
                                   input_sequence_length=input_sequence_length,
                                   target_sequence_length=target_sequence_length,
                                   min_frequency=0.1, max_frequency=10,
                                   min_amplitude=0.1, max_amplitude=1,
                                   min_offset=-0.5, max_offset=0.5,
                                   num_signals=num_signals, seed=1969)

(x_train, _), y_train = next(train_data_generator)

y_train_predicted = predict(x_train, encoder_predict_model, decoder_predict_model, num_steps_to_predict)

# Select 10 random examples to plot
indices = np.random.choice(range(x_train.shape[0]), replace=False, size=10)

for index in indices:
    plot_prediction(x_train[index, :, :], y_train[index, :, :], y_train_predicted[index, :, :])


# ## Next steps & Discussion
#
# There are many things that could be done to either extend or improve this model. Here are a few ideas.
#
# * There's no reason why the encoder and decoder should have the same complexity or the same number of layers. As well as doing a simple hyper parameter search, it could be interesting to implement a model with different encoder and decoder sizes. To do this, one would have to add a dense layer after retrieving the states of the encoder to transform them into the correct size.
# * Encapsulate the encoder-decoder by creating a class with a fit/predict interface. This is actually something I have done, it's extremely useful as it allows to instantiate seq2seq models as easily as one would instantiate a scikit learn model.
# * Add the ability to add context vectors to the state output by the encoder. The encoder is able to produce an input vector for the decoder based on the time series. It is possible to add constant features to the model by duplicating them at each input timestep. However, adding the ability to extend the encoder output state with a constant vector that represents context might also be a good idea (for example, if you're predicting the evolution of housing prices, you might want to tell your model which geographical area you are in, since prices might not evolve in the same manner depending on location). This is not the attention mechanism often used in NLP that also produces what is called a context vector(a context vector that is updated at each step of the decoder). But since adding attention to NLP seq2seq applications has hugely improved state of the art. It might also be worth looking into attention for sequence prediction.
# * As described above, study how the encoder creates a representation of the input sequence by looking at the state vector.
# * It appears that our model struggles on signals that have low frequency, one explanation might be that the model must "see" at least a certain number of periods to determine the frequency of the signal. An interesting questions to answer might be: How many periods of the constituent signals are required for the model to be accurate?
# * Although our model was only train on an output sequence of length 15, it appears to be able to predict beyond that limit, this is something we can exploit with the prediction models.

# ### Thanks for reading :)
#
# I welcome questions or comments, you can find me on LinkedIn.
#
# Author: Luke Tonin
# LinkedIn: https://fr.linkedin.com/in/luketonin
# Github: https://github.com/LukeTonin/
#

# In[15]:


# Let's convert this notebook to a README for the GitHub project's title page:

get_ipython().system('jupyter nbconvert --to markdown keras-seq2seq-signal-prediction.ipynb')
get_ipython().system('mv keras-seq2seq-signal-prediction.md README.md')

