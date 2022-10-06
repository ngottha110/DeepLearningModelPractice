import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Input,MaxPool2D,Dropout
from tensorflow.keras.models import Model

class AlexNet(tf.keras.Model):
  def __init__(self, kernel_size, filters, pad, stride, pool, drop_out, active ='relu'):
    super(AlexNet, self).__init__(name='')
    filters1, filters2, filters3,filters4,filters5,filter6,filter7,filter8 = filters
    kernel_size1, kernel_size2, kernel_size3, kernel_size4,kernel_size5 = kernel_size
    pad1 , pad2, pad3 = pad
    stride1,stride2,stride3,stride4,stride5,stride6,stride7,stride8 = stride
    pool1, pool2, pool3 = pool
    drop_out1, drop_out2 = drop_out

    self.conv2a = tf.keras.layers.Conv2D(filters1, kernel_size1,pad1 , stride1)
    self.maxp2a = tf.keras.layers.MaxPool2D(pool1, stride2)

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size2, stride3, pad2, activation = active)
    self.maxp2b = tf.keras.layers.MaxPool2D(pool2, stride4)

    self.conv2c = tf.keras.layers.Conv2D(filters3, kernel_size3, stride5, pad3, activation = active)
    self.conv2d = tf.keras.layers.Conv2D(filters4, kernel_size4, stride6, activation = active)

    self.conv2e = tf.keras.layers.Conv2D(filters5, kernel_size5 , stride7)
    self.maxp2c = tf.keras.layers.MaxPool2D(pool3, stride8)

    self.fc1 = tf.keras.layers.Dense(filter6, activation = active)
    self.do1 = tf.keras.layers.Dropout(drop_out1)

    self.fc2 = tf.keras.layers.Dense(filter7, activation = active)
    self.do2 = tf.keras.layers.Dropout(drop_out2)

    self.fc3 = tf.keras.layers.Dense(filter8)

def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.maxp2a(x)

    x = self.conv2b(x)
    x = self.maxp2b(x)

    x = self.conv2c(x)
    x = self.conv2d(x)

    x = self.conv2e(x)
    x = self.maxp2c(x)

    x = self.fc1(x)
    x = self.do1(x)

    x = self.fc2(x)
    x = self.do2(x)

    x = self.fc3(x)
    return x