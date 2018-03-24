from keras.engine import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.layers import RNN
from keras.layers.recurrent import _generate_dropout_mask, _generate_dropout_ones

import warnings

#import tensorflow as tf
#import tensorflow.contrib.rnn as rnn

class NASCell(Layer):
    """Neural Architecture Search (NAS) recurrent network cell.

    This implements the recurrent cell from the paper:
    https://arxiv.org/abs/1611.01578
    Barret Zoph and Quoc V. Le.
    "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

    The class uses an optional projection layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        projection_units: (optional) Positive integer, The output dimensionality
            for the projection matrices.  If None, no projection is performed.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        projection_activation: Activation function to use
            for the projection step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        projection_initializer: Initializer for the `projection_kernel`
            weights matrix,
            used for the linear transformation of the projection step.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        projection_regularizer: Regularizer function applied to
            the `projection_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        projection_constraint: Constraint function applied to
            the `projection_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 projection_units=None,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 projection_activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 projection_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 projection_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 **kwargs):
        super(NASCell, self).__init__(**kwargs)
        self.units = units
        self.projection_units = projection_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.projection_activation = activations.get(projection_activation)
        self.cell_activation = activations.get('relu')
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.projection_initializer = initializers.get(projection_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.projection_constraint = constraints.get(projection_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation

        if self.projection_units is not None:
            self.state_size = (self.projection_units, self.units)
        else:
            self.state_size = (self.units, self.units)

        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.projection_units is not None:
            recurrent_output_dim = self.projection_units
        else:
            recurrent_output_dim = self.units

        self.kernel = self.add_weight(shape=(input_dim, self.units * 8),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(recurrent_output_dim, self.units * 8),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.projection_units is not None:
            self.projection_kernel = self.add_weight(
                shape=(self.units, self.projection_units),
                name='projection_kernel',
                initializer=self.projection_initializer,
                regularizer=self.projection_regularizer,
                constraint=self.projection_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 6,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 8,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_0 = self.kernel[:, :self.units]
        self.kernel_1 = self.kernel[:, self.units: self.units * 2]
        self.kernel_2 = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_3 = self.kernel[:, self.units * 3: self.units * 4]
        self.kernel_4 = self.kernel[:, self.units * 4: self.units * 5]
        self.kernel_5 = self.kernel[:, self.units * 5: self.units * 6]
        self.kernel_6 = self.kernel[:, self.units * 6: self.units * 7]
        self.kernel_7 = self.kernel[:, self.units * 7:]

        self.recurrent_kernel_0 = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_1 = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_2 = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_3 = self.recurrent_kernel[:, self.units * 3: self.units * 4]
        self.recurrent_kernel_4 = self.recurrent_kernel[:, self.units * 4: self.units * 5]
        self.recurrent_kernel_5 = self.recurrent_kernel[:, self.units * 5: self.units * 6]
        self.recurrent_kernel_6 = self.recurrent_kernel[:, self.units * 6: self.units * 7]
        self.recurrent_kernel_7 = self.recurrent_kernel[:, self.units * 7:]

        if self.use_bias:
            self.bias_0 = self.bias[:self.units]
            self.bias_1 = self.bias[self.units: self.units * 2]
            self.bias_2 = self.bias[self.units * 2: self.units * 3]
            self.bias_3 = self.bias[self.units * 3: self.units * 4]
            self.bias_4 = self.bias[self.units * 4: self.units * 5]
            self.bias_5 = self.bias[self.units * 5: self.units * 6]
            self.bias_6 = self.bias[self.units * 6: self.units * 7]
            self.bias_7 = self.bias[self.units * 7:]
        else:
            self.bias_0 = None
            self.bias_1 = None
            self.bias_2 = None
            self.bias_3 = None
            self.bias_4 = None
            self.bias_5 = None
            self.bias_6 = None
            self.bias_7 = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=8)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            _recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=8)
            self._recurrent_dropout_mask = _recurrent_dropout_mask

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_0 = inputs * dp_mask[0]
                inputs_1 = inputs * dp_mask[1]
                inputs_2 = inputs * dp_mask[2]
                inputs_3 = inputs * dp_mask[3]
                inputs_4 = inputs * dp_mask[4]
                inputs_5 = inputs * dp_mask[5]
                inputs_6 = inputs * dp_mask[6]
                inputs_7 = inputs * dp_mask[7]
            else:
                inputs_0 = inputs
                inputs_1 = inputs
                inputs_2 = inputs
                inputs_3 = inputs
                inputs_4 = inputs
                inputs_5 = inputs
                inputs_6 = inputs
                inputs_7 = inputs

            x_0 = K.dot(inputs_0, self.kernel_0)
            x_1 = K.dot(inputs_1, self.kernel_1)
            x_2 = K.dot(inputs_2, self.kernel_2)
            x_3 = K.dot(inputs_3, self.kernel_3)
            x_4 = K.dot(inputs_4, self.kernel_4)
            x_5 = K.dot(inputs_5, self.kernel_5)
            x_6 = K.dot(inputs_6, self.kernel_6)
            x_7 = K.dot(inputs_7, self.kernel_7)

            if self.use_bias:
                x_0 = K.bias_add(x_0, self.bias_0)
                x_1 = K.bias_add(x_1, self.bias_1)
                x_2 = K.bias_add(x_2, self.bias_2)
                x_3 = K.bias_add(x_3, self.bias_3)
                x_4 = K.bias_add(x_4, self.bias_4)
                x_5 = K.bias_add(x_5, self.bias_5)
                x_6 = K.bias_add(x_6, self.bias_6)
                x_7 = K.bias_add(x_7, self.bias_7)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_0 = h_tm1 * rec_dp_mask[0]
                h_tm1_1 = h_tm1 * rec_dp_mask[1]
                h_tm1_2 = h_tm1 * rec_dp_mask[2]
                h_tm1_3 = h_tm1 * rec_dp_mask[3]
                h_tm1_4 = h_tm1 * rec_dp_mask[4]
                h_tm1_5 = h_tm1 * rec_dp_mask[5]
                h_tm1_6 = h_tm1 * rec_dp_mask[6]
                h_tm1_7 = h_tm1 * rec_dp_mask[7]
            else:
                h_tm1_0 = h_tm1
                h_tm1_1 = h_tm1
                h_tm1_2 = h_tm1
                h_tm1_3 = h_tm1
                h_tm1_4 = h_tm1
                h_tm1_5 = h_tm1
                h_tm1_6 = h_tm1
                h_tm1_7 = h_tm1

            # First Layer
            layer1_0 = self.recurrent_activation(x_0 + K.dot(h_tm1_0, self.recurrent_kernel_0))
            layer1_1 = self.cell_activation(x_1 + K.dot(h_tm1_1, self.recurrent_kernel_1))
            layer1_2 = self.recurrent_activation(x_2 + K.dot(h_tm1_2, self.recurrent_kernel_2))
            layer1_3 = self.cell_activation(x_3 * K.dot(h_tm1_3, self.recurrent_kernel_3))
            layer1_4 = self.activation(x_4 + K.dot(h_tm1_4, self.recurrent_kernel_4))
            layer1_5 = self.recurrent_activation(x_5 + K.dot(h_tm1_5, self.recurrent_kernel_5))
            layer1_6 = self.activation(x_6 + K.dot(h_tm1_6, self.recurrent_kernel_6))
            layer1_7 = self.recurrent_activation(x_7 + K.dot(h_tm1_7, self.recurrent_kernel_7))

            # Second Layer
            layer2_0 = self.activation(layer1_0 * layer1_1)
            layer2_1 = self.activation(layer1_2 + layer1_3)
            layer2_2 = self.activation(layer1_4 * layer1_5)
            layer2_3 = self.recurrent_activation(layer1_6 + layer1_7)

            # Inject the Cell
            layer2_0 = self.activation(layer2_0 + c_tm1)

            # Third Layer
            layer3_0_pre = layer2_0 * layer2_1
            c = layer3_0_pre  # create a new cell
            layer3_0 = layer3_0_pre
            layer3_1 = self.activation(layer2_2 + layer2_3)

            # Final Layer
            h = self.activation(layer3_0 * layer3_1)

            if self.projection_units is not None:
                h = self.projection_activation(K.dot(h, self.projection_kernel))

        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            zr = K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                zr = K.bias_add(zr, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units: 4 * self.units]
            z4 = z[:, 4 * self.units: 5 * self.units]
            z5 = z[:, 5 * self.units: 6 * self.units]
            z6 = z[:, 6 * self.units: 7 * self.units]
            z7 = z[:, 7 * self.units:]

            zr0 = zr[:, :self.units]
            zr1 = zr[:, self.units: 2 * self.units]
            zr2 = zr[:, 2 * self.units: 3 * self.units]
            zr3 = zr[:, 3 * self.units: 4 * self.units]
            zr4 = zr[:, 4 * self.units: 5 * self.units]
            zr5 = zr[:, 5 * self.units: 6 * self.units]
            zr6 = zr[:, 6 * self.units: 7 * self.units]
            zr7 = zr[:, 7 * self.units:]

            # First Layer
            layer1_0 = self.recurrent_activation(z0 + zr0)
            layer1_1 = self.cell_activation(z1 + zr1)
            layer1_2 = self.recurrent_activation(z2 + zr2)
            layer1_3 = self.cell_activation(z3 * zr3)
            layer1_4 = self.activation(z4 + zr4)
            layer1_5 = self.recurrent_activation(z5 + zr5)
            layer1_6 = self.activation(z6 + zr6)
            layer1_7 = self.recurrent_activation(z7 + zr7)

            # Second Layer
            layer2_0 = self.activation(layer1_0 * layer1_1)
            layer2_1 = self.activation(layer1_2 + layer1_3)
            layer2_2 = self.activation(layer1_4 * layer1_5)
            layer2_3 = self.recurrent_activation(layer1_6 + layer1_7)

            # Inject the Cell
            layer2_0 = self.activation(layer2_0 + c_tm1)

            # Third Layer
            layer3_0_pre = layer2_0 * layer2_1
            c = layer3_0_pre
            layer3_0 = layer3_0_pre
            layer3_1 = self.activation(layer2_2 + layer2_3)

            # Final Layer
            h = self.activation(layer3_0 * layer3_1)

            if self.projection_units is not None:
                h = self.projection_activation(K.dot(h, self.projection_kernel))

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'projection_units': self.projection_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'projection_activation': activations.serialize(self.projection_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'projection_initializer': initializers.serialize(self.projection_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'projection_regularizer': regularizers.serialize(self.projection_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'projection_constraint': constraints.serialize(self.projection_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(NASCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NASRNN(RNN):
    """Neural Architecture Search (NAS) recurrent network cell.

    This implements the recurrent cell from the paper:
    https://arxiv.org/abs/1611.01578
    Barret Zoph and Quoc V. Le.
    "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

    The class uses an optional projection layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        projection_units: (optional) Positive integer, The output dimensionality
            for the projection matrices.  If None, no projection is performed.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        projection_activation: Activation function to use
            for the projection step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        projection_initializer: Initializer for the `projection_kernel`
            weights matrix,
            used for the linear transformation of the projection step.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        projection_regularizer: Regularizer function applied to
            the `projection_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        projection_constraint: Constraint function applied to
            the `projection_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with NestedLSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Nested LSTMs](https://arxiv.org/abs/1801.10308)
    """

    def __init__(self, units,
                 projection_units=None,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 projection_activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 projection_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 projection_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=2`.'
                          'Please update your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = NASCell(units, projection_units,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       projection_activation=projection_activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       projection_initializer=projection_initializer,
                       unit_forget_bias=unit_forget_bias,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       projection_regularizer=projection_regularizer,
                       kernel_constraint=kernel_constraint,
                       recurrent_constraint=recurrent_constraint,
                       bias_constraint=bias_constraint,
                       projection_constraint=projection_constraint,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       implementation=implementation)
        super(NASRNN, self).__init__(cell,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      go_backwards=go_backwards,
                                      stateful=stateful,
                                      unroll=unroll,
                                      **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(NASRNN, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state,
                                            constants=constants)

    @property
    def units(self):
        return self.cell.units

    @property
    def projection_units(self):
        return self.cell.projection_units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def projection_activation(self):
        return self.cell.projection_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def projection_initializer(self):
        return self.cell.projection_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def projection_regularizer(self):
        return self.cell.projection_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def projection_constraint(self):
        return self.cell.projection_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'projection_units': self.projection_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'projection_activation': activations.serialize(self.projection_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'projection_initializer': initializers.serialize(self.projection_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'projection_regularizer': regularizers.serialize(self.projection_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'projection_constraint': constraints.serialize(self.projection_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(NASRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 2
        return cls(**config)
