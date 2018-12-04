"""
ResNet model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
import os
from . import imagenet_utils
import keras.layers as layers
import keras.models as models

# I use the 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# thus, change mode to 'tf', nerver mind , let's try origin mode caffe in keras source code....
# download_link : https://github.com/fchollet/deep-learning-models/releases
def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='caffe')

def squeeze_excitation_layer(x, out_dim, stage, block, ratio=16):
    '''
        SE channel attention
        input:
            x
            out_dim : channel default
            ratio : reduction rate, defualt 16
    '''
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(out_dim//ratio, activation='relu', name='se_dense1_'+str(stage)+block)(squeeze)
    excitation = layers.Dense(out_dim, activation='sigmoid', name='se_dense2_'+str(stage)+block)(excitation)
    excitation = layers.Reshape((1, 1, out_dim))(excitation)

    scale = layers.multiply([x, excitation])

    return scale

def identity_block(input_tensor, kernel_size, filters, stage, block, bn_axis, bottleneck=True):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        bottleneck : if res50 or res>50 than True else False
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if bottleneck:
        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    else:
        x = layers.Conv2D(filters1, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters1, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    scaled_x = squeeze_excitation_layer(x, int(x.shape[-1]), stage, block)
    x = layers.add([scaled_x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               bn_axis,
               strides=(2, 2),
               bottleneck=True):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        bottleneck : if res50 or res>50 than True else False
    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if bottleneck:
        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)
    else:
        x = layers.Conv2D(filters1, kernel_size, strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters1, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

        shortcut = layers.Conv2D(filters1, (1, 1), strides=strides,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

    scaled_x = squeeze_excitation_layer(x, int(x.shape[-1]), stage, block)
    x = layers.add([scaled_x, shortcut])
    x = layers.Activation('relu')(x)
    return x

class SE_ResNet(object):

    def build(self,
              input_shape=None,
              data_format='channels_last',
              pooling=None,
              repetitions=None,
              bottleneck=True,
              model_name='resnet50'):

        assert len(repetitions) == 4

        img_input = layers.Input(shape=input_shape)

        if data_format == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3

        # stage1
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, (7, 7),
                         strides=(2, 2),
                         padding='valid',
                         kernel_initializer='he_normal',
                         name='conv1_4' if input_shape[2] == 4 else 'conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        block_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        base_num1 = 64
        base_num2 = 256

        # loop for stages from stage2 to ...
        for i, num in enumerate(repetitions):
            stage = i+2
            # fisrt block in i-th stage
            x = conv_block(x, 3, [base_num1, base_num1, base_num2],
                           stage=stage,
                           block='a',
                           bn_axis=bn_axis,
                           strides=(1, 1) if i == 0 else (2, 2),
                           bottleneck=bottleneck)
            # loop for rest block in i-th stage
            for ii in range(1, num):
                #block_name = block_names[ii+1]
                x = identity_block(x, 3, [base_num1, base_num1, base_num2],
                                   stage=stage,
                                   block='b'+str(ii) if (repetitions[2] > 6 and stage in [3,4]) else block_names[ii],    #res50（包括50）以下按照keras内置的命名格式，res50以上则按照权值文件的命名格式，参考：https://github.com/GKalliatakis/Keras-Application-Zoo/blob/master/resnet101.py
                                   bn_axis=bn_axis,
                                   bottleneck=bottleneck)
            base_num1 = 2 * base_num1
            base_num2 = 2 * base_num2

        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        elif pooling:
            raise('no such pooling type!!')

        # create model
        model = models.Model(img_input, x, name=model_name)

        return model

    def se_resnet18(self,
                 input_shape,
                 data_format='channels_last',
                 pooling=None,
                 weights=None):
        model = self.build(input_shape,
                           data_format=data_format,
                           pooling=pooling,
                           repetitions=[2, 2, 2, 2],
                           model_name='resnet18',
                           bottleneck=False)
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights)
        return model

    def se_resnet34(self,
                 input_shape,
                 data_format='channels_last',
                 pooling=None,
                 weights=None):
        model = self.build(input_shape,
                           data_format=data_format,
                           pooling=pooling,
                           repetitions=[3, 4, 6, 3],
                           model_name='resnet34',
                           bottleneck=False)
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights)
        return model

    def se_resnet50(self,
                 input_shape,
                 data_format='channels_last',
                 pooling=None,
                 weights=None):
        model = self.build(input_shape,
                           data_format=data_format,
                           pooling=pooling,
                           repetitions=[3, 4, 6, 3],
                           model_name='resnet50')
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights, by_name=True if input_shape[2] == 4 else False)
        return model

    def se_resnet101(self,
                  input_shape,
                  data_format='channels_last',
                  pooling=None,
                  weights=None):
        model = self.build(input_shape,
                           data_format=data_format,
                           pooling=pooling,
                           repetitions=[3, 4, 23, 3],
                           model_name='resnet101')
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights)
        return model

    def se_resnet152(self,
                  input_shape,
                  data_format='channels_last',
                  pooling=None,
                  weights=None):
        model = self.build(input_shape,
                           data_format=data_format,
                           pooling=pooling,
                           repetitions=[3, 8, 36, 3],
                           model_name='resnet152')
        if weights:
            if not os.path.exists(weights):
                raise('no such path {}'.format(weights))
            model.load_weights(weights)
        return model
