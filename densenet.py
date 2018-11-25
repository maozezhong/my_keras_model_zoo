"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
import os
from . import imagenet_utils
import keras.layers as layers
import keras.models as models
import keras.backend as backend

def preprocess_input(x, data_format=None, **kwargs):
    return imagenet_utils.preprocess_input(x)

def dense_block(x, blocks, name, bn_axis):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1), bn_axis=bn_axis)
    return x


def transition_block(x, reduction, name, bn_axis):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name, bn_axis):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

class DenseNet(object):

    def build(self,
                 blocks,
                 input_shape=None,
                 data_format='channels_last',
                 pooling=None):

        img_input = layers.Input(shape=input_shape)

        bn_axis = 3 if data_format == 'channels_last' else 1

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        if input_shape[2] == 4:
            x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_4/conv')(x)
        else:
            x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('relu', name='conv1/relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        x = dense_block(x, blocks[0], name='conv2', bn_axis=bn_axis)
        x = transition_block(x, 0.5, name='pool2', bn_axis=bn_axis)
        x = dense_block(x, blocks[1], name='conv3', bn_axis=bn_axis)
        x = transition_block(x, 0.5, name='pool3', bn_axis=bn_axis)
        x = dense_block(x, blocks[2], name='conv4', bn_axis=bn_axis)
        x = transition_block(x, 0.5, name='pool4', bn_axis=bn_axis)
        x = dense_block(x, blocks[3], name='conv5', bn_axis=bn_axis)

        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)

        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
        elif pooling:
            raise('no such pooling type!!')

        # Create model.
        if blocks == [6, 12, 24, 16]:
            model = models.Model(img_input, x, name='densenet121')
        elif blocks == [6, 12, 32, 32]:
            model = models.Model(img_input, x, name='densenet169')
        elif blocks == [6, 12, 48, 32]:
            model = models.Model(img_input, x, name='densenet201')
        else:
            model = models.Model(img_input, x, name='densenet')

        return model


    def densenet121(self,
                    input_shape,
                    data_format='channels_last',
                    pooling=None,
                    weights=None):

        model = self.build(blocks=[6, 12, 24, 16],
                           input_shape=input_shape,
                           data_format=data_format,
                           pooling=pooling)
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights, by_name=True if input_shape[2] == 4 else False)
        return model


    def densenet169(self,
                    input_shape,
                    data_format='channels_last',
                    pooling=None,
                    weights=None):

        model = self.build(blocks=[6, 12, 32, 32],
                           input_shape=input_shape,
                           data_format=data_format,
                           pooling=pooling)
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights, by_name=True if input_shape[2] == 4 else False)
        return model

    def densenet201(self,
                    input_shape,
                    data_format='channels_last',
                    pooling=None,
                    weights=None):

        model = self.build(blocks=[6, 12, 48, 32],
                           input_shape=input_shape,
                           data_format=data_format,
                           pooling=pooling)
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights, by_name=True if input_shape[2] == 4 else False)
        return model

