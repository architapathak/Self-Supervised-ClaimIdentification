
from keras import backend as K
from keras.layers import Layer,InputSpec,Conv1D 

import tensorflow as tf

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
#         print(self.channels,self.filters_f_g,self.filters_h)

    def build(self, input_shape):
        self.N = input_shape[1]        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

        super(SelfAttention, self).build(input_shape)
        self.input_spec = InputSpec(ndim=3,
                                    axes={2: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        q = Conv1D(filters=self.filters_q_k, kernel_size=3, padding='same')(x)
        k = Conv1D(filters=self.filters_q_k, kernel_size=3, padding='same')(x)
        v = Conv1D(filters=self.filters_v, kernel_size=3, padding='same')(x)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x
#         print('x.shape:',x.shape)
        return [x, beta, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape,self.beta_shape, tuple(self.gamma.shape.as_list())]