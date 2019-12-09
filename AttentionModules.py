
# from keras import backend as K
# from keras.layers import Layer,InputSpec,Conv1D 

# import tensorflow as tf

# class SelfAttention(Layer):
#     def __init__(self, ch, **kwargs):
#         super(SelfAttention, self).__init__(**kwargs)
#         self.channels = ch
#         self.filters_q_k = self.channels // 8
#         self.filters_v = self.channels
# #         print(self.channels,self.filters_f_g,self.filters_h)

#     def build(self, input_shape):
#         self.N = input_shape[1]        
#         self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

#         super(SelfAttention, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True


#     def call(self, x):
#         def hw_flatten(x):
#             return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

#         q = Conv1D(filters=self.filters_q_k, kernel_size=3, padding='same')(x)
#         k = Conv1D(filters=self.filters_q_k, kernel_size=3, padding='same')(x)
#         v = Conv1D(filters=self.filters_v, kernel_size=3, padding='same')(x)
# #         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
#         s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
# #         print('s.shape:',s.shape)
#         beta = K.softmax(s, axis=-1)  # attention map
#         self.beta_shape = tuple(beta.shape[1:].as_list())
# #         print('beta.shape:',beta.shape.as_list())
#         o = K.batch_dot(beta, v)  # [bs, N, C]
# #         print('o.shape:',o.shape)
# #         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
#         x = self.gamma * o + x
# #         print('x.shape:',x.shape)
#         return [x, beta, self.gamma]

#     def compute_output_shape(self, input_shape):
#         return [input_shape,self.beta_shape, tuple(self.gamma.shape.as_list())]

'''
code referred from:
https://arxiv.org/abs/1805.08318 [SAGAN paper]
https://github.com/taki0112/Self-Attention-GAN-Tensorflow/blob/master/SAGAN.py
'''
from keras import backend as K
from keras.layers import Layer,InputSpec
import tensorflow as tf
class SelfAttention_gl(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention_gl, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
        
#         print(kernel_shape_q_k)
#         print(self.channels,self.filters_f_g,self.filters_h)

    def build(self, input_shape):
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1]        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.lam = self.add_weight(name='lam', shape=[1], initializer='ones', trainable=True)
        
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(SelfAttention_gl, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=3,
                                    axes={2: input_shape[-1]})
        self.built = True


    def call(self, x):
        
        q = K.conv1d(x,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]

        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
        x = self.gamma * o + self.lam * x
#         print('x.shape:',x.shape)
        return [x, s, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape,self.beta_shape, tuple(self.gamma.shape.as_list())]
    
class CrossAttention_gl(Layer):
    def __init__(self, ch, **kwargs):
        super(CrossAttention_gl, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
        
#         print(kernel_shape_q_k)
#         print(self.channels,self.filters_f_g,self.filters_h)

    def build(self, input_shape):
#         print('input_shape:',input_shape)
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1]        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.lam = self.add_weight(name='lam', shape=[1], initializer='ones', trainable=True)
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(CrossAttention_gl, self).build(input_shape)
        # Set input spec.
#         self.input_spec = InputSpec(ndim=(3,3),
#                                     axes={2: input_shape[0][-1],2:input_shape[1][-1]})
#         self.built = True


    def call(self, inputs):
        h,x = inputs
        q = K.conv1d(h,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + self.lam * x
    
    
        
#         print('x.shape:',x.shape)
        return [x, s, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape[-1],self.beta_shape, tuple(self.gamma.shape.as_list())] 
    
    
## --------------------------------------exp2------------------------------------------  
class SelfAttention_gl1(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention_gl1, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
        
#         print(kernel_shape_q_k)
#         print(self.channels,self.filters_f_g,self.filters_h)

    def build(self, input_shape):
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1] 
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(SelfAttention_gl1, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=3,
                                    axes={2: input_shape[-1]})
        self.built = True


    def call(self, x):
        
        q = K.conv1d(x,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]

        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
        x = o + x # gammax lam = 1
#         print('x.shape:',x.shape)
        return [x, s]

    def compute_output_shape(self, input_shape):
        return [input_shape,self.beta_shape]
    
class CrossAttention_gl1(Layer):
    def __init__(self, ch, **kwargs):
        super(CrossAttention_gl1, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
        
#         print(kernel_shape_q_k)
#         print(self.channels,self.filters_f_g,self.filters_h)

    def build(self, input_shape):
#         print('input_shape:',input_shape)
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1] 
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(CrossAttention_gl1, self).build(input_shape)
        # Set input spec.
#         self.input_spec = InputSpec(ndim=(3,3),
#                                     axes={2: input_shape[0][-1],2:input_shape[1][-1]})
#         self.built = True


    def call(self, inputs):
        h,x = inputs
        q = K.conv1d(h,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = o + x #gamma, lam=1
    
    
        
#         print('x.shape:',x.shape)
        return [x, s]

    def compute_output_shape(self, input_shape):
        return [input_shape[-1],self.beta_shape] 
#----------------------------------exp3 used in paper---------------
    
class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels

    def build(self, input_shape):
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1]        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(SelfAttention, self).build(input_shape)
        self.input_spec = InputSpec(ndim=3,
                                    axes={2: input_shape[-1]})
        self.built = True


    def call(self, x):
        
        q = K.conv1d(x,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]

        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
        x = self.gamma * o + x
#         print('x.shape:',x.shape)
        return [x, s, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape,self.beta_shape, tuple(self.gamma.shape.as_list())]
    
class CrossAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels

    def build(self, input_shape):
#         print('input_shape:',input_shape)
        kernel_shape_q_k = (1, self.channels, self.filters_q_k)
        kernel_shape_v = (1, self.channels, self.filters_v)
        self.N = input_shape[1]        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        super(CrossAttention, self).build(input_shape)

    def call(self, inputs):
        h,x = inputs
        q = K.conv1d(h,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = tf.matmul(q, k, transpose_b=True)  # # [bs, N, N]
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(beta, v)  # [bs, N, C]
        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x
    
    
        
#         print('x.shape:',x.shape)
        return [x, s, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape[-1],self.beta_shape, tuple(self.gamma.shape.as_list())] 