import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import add_arg_scope, arg_scope, arg_scoped_arguments
import layers as my_layers
from utils import *

SRM_Kernels = np.load('SRM_Kernels.npy')

class YeNet_bl4(Model):
    def conv2d(self,inputs,filters=16,kernel_size=3,strides=1,padding='VALID',activation=None,
                       kernel_initializer=layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.2),
                       kernel_regularizer=layers.l2_regularizer(2e-4),
                       use_bias=True,name="conv"):
      return tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_first",activation=activation,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_initializer=bias_initializer,
                       use_bias=use_bias,name=name)
    def _build_model(self, inputs):
        if self.data_format == 'NCHW': 
            reduction_axis = [2,3]
            concat_axis = 1
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:          
            reduction_axis = [1,2]
            concat_axis = 3
            _inputs = tf.cast(inputs, tf.float32)
        self.inputImage = _inputs
        with arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[2,2], stride=[2,2], padding='VALID',
                       data_format=self.data_format),\
            arg_scope([layers.max_pool2d],
                       kernel_size=[2,2], stride=[2,2], padding='VALID',
                       data_format=self.data_format):
          #arr = np.loadtxt('kernel_0.8_20.txt',dtype=np.float32)
          #arr = np.reshape(arr,[1,20,8,8])
          #arr = np.transpose(arr,(2,3,0,1))
          #print(arr)
          with tf.variable_scope('SRM_preprocess'):
              W_SRM = tf.get_variable('W', initializer=SRM_Kernels, \
                             dtype=tf.float32, \
                             regularizer=None)
              conv = tf.nn.conv2d(_inputs,W_SRM, [1,1,1,1], 'VALID',data_format=self.data_format)
              actv = tf.clip_by_value(conv,-3,3)
          with tf.variable_scope('Layer2'): 
              conv1=self.conv2d(actv,filters=30,name="conv1")
              res=tf.nn.relu(conv1)
          with tf.variable_scope('Layer3'):  
              conv1=self.conv2d(res,filters=30,name="conv1")
              res=tf.nn.relu(conv1)
          with tf.variable_scope('Layer4'): 
              conv1=self.conv2d(res,filters=30,name="conv1")
              actv1=layers.avg_pool2d(conv1)
              res=tf.nn.relu(actv1)
          with tf.variable_scope('Layer5'): 
              conv1=self.conv2d(res,filters=32,kernel_size=5,name="conv1")
              actv1=layers.avg_pool2d(conv1,kernel_size=[3,3])
              res=tf.nn.relu(actv1)
          with tf.variable_scope('Layer6'): 
              conv1=self.conv2d(res,filters=32,kernel_size=5,name="conv1")
              actv1=layers.avg_pool2d(conv1,kernel_size=[3,3])
              res=tf.nn.relu(actv1)
          with tf.variable_scope('Layer7'): 
              conv1=self.conv2d(res,filters=32,kernel_size=5,name="conv1")
              actv1=layers.avg_pool2d(conv1,kernel_size=[3,3])
              res=tf.nn.relu(actv1)
          with tf.variable_scope('Layer8'): 
              conv1=self.conv2d(res,filters=16,kernel_size=3,name="conv1")
              res=tf.nn.relu(conv1)
          with tf.variable_scope('Layer9'): 
              conv1=self.conv2d(res,filters=16,kernel_size=3,strides=3,name="conv1")
              res=tf.nn.relu(conv1)
          with tf.variable_scope('ensemble'):
              conv1=self.conv2d(res,filters=16,kernel_size=3,strides=1,padding='SAME',name="conv1")
              actv1=tf.nn.relu(conv1)
              conv2=self.conv2d(res,filters=16,kernel_size=3,strides=1,padding='SAME',name="conv2")
              actv2=tf.nn.sigmoid(conv2)
              conv3=self.conv2d(res,filters=16,kernel_size=3,strides=1,padding='SAME',name="conv3")
              actv3=tf.nn.tanh(conv3)
              conv4=self.conv2d(res,filters=16,kernel_size=3,strides=1,padding='SAME',name="conv4")
              actv4=tf.nn.elu(conv4)
          with tf.variable_scope('combine'): 
              ip1 = layers.fully_connected(layers.flatten(actv1),num_outputs=2,activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                biases_initializer=tf.constant_initializer(0.), scope='ip1')
              ip2 = layers.fully_connected(layers.flatten(actv2),num_outputs=2,activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                biases_initializer=tf.constant_initializer(0.), scope='ip2')
              ip3 = layers.fully_connected(layers.flatten(actv3),num_outputs=2,activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                biases_initializer=tf.constant_initializer(0.), scope='ip3')
              ip4 = layers.fully_connected(layers.flatten(actv4),num_outputs=2,activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                biases_initializer=tf.constant_initializer(0.), scope='ip4')
              combine = tf.concat([ip1,ip2,ip3,ip4],1)
              ip = layers.fully_connected(layers.flatten(combine),num_outputs=2,activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
