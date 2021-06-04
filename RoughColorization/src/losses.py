from keras import backend as K

def coloringLoss_OneAccuracy(y_true, y_pred):
    shape = K.shape(y_true)
    h = K.reshape(shape[1], (1,1))
    w = K.reshape(shape[2], (1,1))
    denom = 1 / K.cast(K.reshape(K.dot(h, w), (1,1)), dtype = 'float32')
    return K.dot(K.reshape(K.sum(K.cast(K.less_equal(K.abs(y_true - y_pred), 1), dtype = 'float32')), (1,1)), denom)

def coloringLoss_ThreeAccuracy(y_true, y_pred):
    shape = K.shape(y_true)
    h = K.reshape(shape[1], (1,1))
    w = K.reshape(shape[2], (1,1))
    denom = 1 / K.cast(K.reshape(K.dot(h, w), (1,1)), dtype = 'float32')
    return K.dot(K.reshape(K.sum(K.cast(K.less_equal(K.abs(y_true - y_pred), 3), dtype = 'float32')), (1,1)), denom)

def coloringLoss_OneAccuracyYUV(y_true, y_pred):
    V_acc=coloringLoss_OneAccuracy(y_true[:,:,:,0], y_pred[:,:,:,0])
    U_acc=coloringLoss_OneAccuracy(y_true[:,:,:,1], y_pred[:,:,:,1])
    Y_acc=coloringLoss_OneAccuracy(y_true[:,:,:,2], y_pred[:,:,:,2])

    return (V_acc+U_acc)/2.0

def coloringLoss_ThreeAccuracyYUV(y_true, y_pred):
    V_acc=coloringLoss_ThreeAccuracy(y_true[:,:,:,0], y_pred[:,:,:,0])
    U_acc=coloringLoss_ThreeAccuracy(y_true[:,:,:,1], y_pred[:,:,:,1])
    Y_acc=coloringLoss_ThreeAccuracy(y_true[:,:,:,2], y_pred[:,:,:,2])

    return (V_acc+U_acc)/2.0