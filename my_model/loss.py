import imp
import numpy as np
import tensorflow as tf
from const import EPSILON, LAMBDA_COORD, LAMBDA_LOC
from utils import basicIOU

def yolo_loss(y_pred=None, y_true=None):
    pred_obj_conf = y_pred[:, 0:1]
    pred_box_coord = y_pred[:, 1:5]
    print(f'pred_obj_conf_shape: {pred_obj_conf.shape}')
    print(f'pred_box_coord_shape: {pred_box_coord.shape}')

    target_obj_conf = y_true[:, :1]
    true_box_coord = y_true[:, 1:5]
    print(f'target_obj_conf_shape: {target_obj_conf.shape}')
    print(f'true_box_coord_shape: {true_box_coord.shape}')

    pred_box_offset_coord = tf.stack([pred_box_coord[..., 0:1], # x_hat
                                        pred_box_coord[..., 1:2],   # y_hat
                                        tf.sign(pred_box_coord[..., 2:3]) * tf.sqrt(pred_box_coord[..., 2:3] + EPSILON),    # sqrt(w_hat)
                                        tf.sign(pred_box_coord[..., 2:3]) * tf.sqrt(pred_box_coord[..., 3:4] + EPSILON)],   # sqrt(h_hat)
                                        axis=-1)
    print(f'pred_box_offset_coord_shape: {pred_box_offset_coord.shape}')

    target_box_offset_coord = tf.stack([true_box_coord[..., 0:1],
                                        true_box_coord[..., 1:2],
                                        tf.sign(true_box_coord[..., 2:3]) * tf.sqrt(true_box_coord[..., 2:3] + EPSILON),
                                        tf.sign(true_box_coord[..., 3:4]) * tf.sqrt(true_box_coord[..., 3:4] + EPSILON)],
                                        axis=-1)
    print(f'target_box_offset_coord_shape: {target_box_offset_coord.shape}')

    # computing the IOU loss
    pred_ious = basicIOU(pred_box_coord, true_box_coord)
    predictor_mask = tf.reduce_max(pred_ious, axis=1, keepdims=True)
    predictor_mask = tf.cast(pred_ious>=predictor_mask, tf.float32) * target_obj_conf
    noobj_mask = tf.ones_like(target_obj_conf) - target_obj_conf
    print(f'pred_ious_shape: {pred_ious.shape}')
    print(f'predictor_mask_shape: {predictor_mask.shape}')
    print(f'noobj_mask_shape: {noobj_mask.shape}')

    # computing the location loss
    predictor_mask = predictor_mask[:,:,None]
    loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask*(target_box_offset_coord - pred_box_offset_coord)), axis=[1,2]))
    print(f'predictor_mask_shape: {predictor_mask.shape}')
    print(f'loc_loss_shape: {loc_loss.shape}')

    # computing the confidence loss
    obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_obj_conf*(1 - pred_obj_conf)), axis=[-1]))
    noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask*(0 - pred_obj_conf)), axis=[-1]))
    print(f'obj_loss_shape: {obj_loss.shape}')
    print(f'noobj_loss_shape: {noobj_loss.shape}')

    # computing final loss
    loss = (LAMBDA_LOC * loc_loss) + obj_loss + (0.1 * noobj_loss)
    print(f'loss_shape: {loss.shape}')

    return tf.reduce_sum(loss)





if __name__ == '__main__':
    
    # y_pred = tf.convert_to_tensor([[1, 2, 3, 2, 4], [0, 3, 8.5, 2, 3], [1, 2, 3, 2, 4]], dtype=tf.float32)
    # y_true = tf.convert_to_tensor([[1, 4, 5, 4, 2], [1, 2, 3, 2, 4], [1, 6.5, 3.5, 3, 3]], dtype=tf.float32)
    # loss = yolo_loss(y_pred, y_true)
    # print(f'loss = {loss}')
    pass
