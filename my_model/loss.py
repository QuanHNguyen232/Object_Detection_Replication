import numpy as np
import tensorflow as tf
import const

def yolo_loss(y_pred=None, y_true=None):
    pred_obj_conf = y_pred[:, 0:1]
    pred_box_coord = y_pred[:, 1:5]
    
    target_obj_conf = y_true[:, :1]
    true_box_coord = y_true[:, 1:5]

    noobj_mask = tf.ones_like(target_obj_conf) - target_obj_conf

    # pred_box_offset_coord = tf.stack([pred_box_coord[:,0:1],
    #                                     pred_box_coord[:,1:2],
    #                                     tf.sqrt(pred_box_coord[:,2:3]),
    #                                     tf.sqrt(pred_box_coord[:,3:4])], axis=-1)

    # target_box_offset_coord = tf.stack([true_box_coord[:,0:1],
    #                                     true_box_coord[:,1:2],
    #                                     tf.sqrt(true_box_coord[:,2:3]),
    #                                     tf.sqrt(true_box_coord[:,3:4])], axis=-1)

    # loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_obj_conf*(target_box_offset_coord - pred_box_offset_coord)), axis=[-1]))

    obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_obj_conf*(1 - pred_obj_conf)), axis=[-1]))
    noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask*(0 - pred_obj_conf)), axis=[-1]))

    # loss = obj_loss + 0.1*noobj_loss + const.LAMBDA_LOC*loc_loss
    loss = obj_loss + 0.1*noobj_loss + const.LAMBDA_LOC*0
    
    return tf.reduce_sum(loss)


if __name__ == '__main__':
    
    a = np.zeros(dtype=np.float32, shape=(None, 5))
    print(a.shape)