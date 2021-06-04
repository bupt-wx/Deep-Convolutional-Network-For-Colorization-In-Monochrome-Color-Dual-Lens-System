from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv3D, Conv2DTranspose
from conv3dTranspose import Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras import backend as K
from keras.layers import Input, Add, add, multiply
from keras.layers.core import Lambda, Permute, Reshape
from ipykernel import kernelapp as app
import tensorflow as tf
import numpy as np


def _resNetBlock_(filters, ksize, stride, padding, act_func):
    conv1 = Conv2D(filters, ksize, strides = stride, padding = padding)
    bn1 = BatchNormalization(axis = -1)
    act1 = Activation(act_func)
    conv2 = Conv2D(filters,ksize, strides = stride, padding = padding)
    bn2 = BatchNormalization(axis = -1)
    act2 = Activation(act_func)
    add = Add()
    return [conv1, bn1, act1, conv2, bn2, act2, add]

def _getConcatVolume_(inputs):
    left_tensor, right_tensor = inputs
    concat_data = []
    tmp_data = K.concatenate([left_tensor, right_tensor], axis = 3)
    concat_data.append(tmp_data)
    return concat_data

def _getChannels_(inputs):
    channel1 = inputs[:,:,:,0:1]
    channel2 = inputs[:,:,:,1:2]
    return [channel1,channel2]

def  _computeLinearScore_(cv, d):
    cv = K.permute_dimensions(cv, (0,2,3,1))
    disp_map = K.reshape(K.arange(0, d, dtype = K.floatx()), (1,1,d,1))
    output = K.conv2d(cv, disp_map, strides = (1,1), padding = 'valid')
    return K.squeeze(output, axis = -1)

def _computeSoftArgMin_(cv, rough_img):
    return cv+rough_img

def getOutputFunction(output):
        if output == 'linear':
                return _computeLinearScore_
        if output == 'softargmin':
                return _computeSoftArgMin_

def _createUniFeatureGuidance_(input_shape, num_res, filters, first_ksize, ksize, act_func, padding):
    conv1 = Conv2D(filters, first_ksize, strides = 1, padding = padding, input_shape = input_shape)
    bn1 = BatchNormalization(axis = -1)
    act1 = Activation(act_func)
    layers = [conv1, bn1, act1]
    for i in range(num_res):
        layers += _resNetBlock_(filters, ksize, 1, padding, act_func)

    filters = 1
    output = Conv2D(filters, ksize, strides = 1, padding = padding)
    layers.append(output)
    return layers

def _createUniFeatureFinal_(input_shape, num_res, filters, first_ksize, ksize, act_func, padding):
    conv1 = Conv2D(filters, first_ksize, strides = 1, padding = padding, input_shape = input_shape)
    bn1 = BatchNormalization(axis = -1)
    act1 = Activation(act_func)
    layers = [conv1, bn1, act1]
    for i in range(num_res):
        layers += _resNetBlock_(filters, ksize, 1, padding, act_func)

    filters = 1
    output = Conv2D(filters, ksize, strides = 1, padding = padding)
    layers.append(output)
    return layers

def _createFeatureOutput_(input_shape, act_func, padding):
    filters = 2
    ksize = 1
    conv1 = Conv2D(filters, ksize, strides = 1, padding = padding, input_shape = input_shape)
    act1 = Activation(act_func)
    layers = [conv1, act1]
    return layers

def createFeature(input, layers):
    res = layers[0](input)
    tensor = res
    for layer in layers[1:]:
        if isinstance(layer, Add):
            tensor = layer([tensor, res])
            res = tensor
        else:
            tensor = layer(tensor)
    return tensor

def createColoringNetwork(hp, tp, pre_weight):

    cost_weight = tp['cost_volume_weight_path']
    first_ksize = hp['first_kernel_size']
    ksize = hp['kernel_size']
    num_filters = hp['base_num_filters']
    act_func = hp['act_func']
    output = hp['output']
    num_res = hp['num_res']
    padding = hp['padding']
    K.set_image_data_format(hp['data_format'])

    input_shape_rough = (None, None, 2)
    input_shape_guidance = (None, None, 1)
    rough_img = Input(input_shape_rough, dtype = "float32")
    guidance_img = Input(input_shape_guidance, dtype = "float32")

    layers = _createUniFeatureGuidance_(input_shape_guidance, num_res, num_filters, first_ksize, ksize, act_func, padding)
    guidance_feature = createFeature(guidance_img, layers)

    rough_img_channel1,rough_img_channel2 = Lambda(_getChannels_)(rough_img)

    unifeatures1 = [rough_img_channel1, guidance_feature]
    cv1 = Lambda(_getConcatVolume_)(unifeatures1)
    unifeatures2 = [rough_img_channel2, guidance_feature]
    cv2 = Lambda(_getConcatVolume_)(unifeatures2)

    cv_shape = (None, None, 2)

    layers2 = _createUniFeatureFinal_(cv_shape, num_res, num_filters, first_ksize, ksize, act_func, padding)
    corrected_residue_channel1 = createFeature(cv1, layers2)  
    corrected_residue_channel2 = createFeature(cv2, layers2)

    uniresidue = [corrected_residue_channel1, corrected_residue_channel2]
    concat_result = Lambda(_getConcatVolume_)(uniresidue)
    
    output_shape = (None, None, 2)
    layers_output = _createFeatureOutput_(output_shape, act_func, padding)
    corrected_residue = createFeature(concat_result, layers_output) 

    cost_model = Model([rough_img , guidance_img], corrected_residue)

    if pre_weight == 1:
        print("Loading pretrained cost weight...")
        cost_model.load_weights(cost_weight)

    out_func = getOutputFunction(output)
    color_map_residue = Input((None, None, 2))
    output = Lambda(out_func, arguments = {'rough_img':rough_img})(color_map_residue)
    linear_output_model = Model(color_map_residue, output)
    model = Model(cost_model.input, linear_output_model(cost_model.output))

    return model

