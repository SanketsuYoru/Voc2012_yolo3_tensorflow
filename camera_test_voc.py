#utils
import os
import random
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image
import cv2 as cv
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K
from keras.layers import Input, Lambda
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model, Model
#yolo
from yolo import YOLO, detect_video
import yolo_utils
from tensorflow.keras.preprocessing import image
from yolo3.model import yolo_head, yolo_correct_boxes, preprocess_true_boxes, yolo_loss, yolo_body
import os
from yolo3.utils import get_random_data

current_path = os.path.dirname(__file__)

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    为锚框实现非最大值抑制（ Non-max suppression (NMS)）
    
    参数：
        scores - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        boxes - tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小（见下文）
        classes - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        max_boxes - 整数，预测的锚框数量的最大值
        iou_threshold - 实数，交并比阈值。
        
    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类
        
    注意："None"是明显小于max_boxes的，这个函数也会改变scores、boxes、classes的维度，这会为下一步操作提供方便。
    
    """
    # max_boxes_tensor = K.variable(max_boxes,dtype="int32") #用于tf.image.non_max_suppression()
    # # K.get_session().run(K.variable([max_boxes_tensor])) #初始化变量max_boxes_tensor
    
    #使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores,max_boxes,iou_threshold)
    
    #使用K.gather()来选择保留的锚框
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes



def yolo_filter_boxes(box_confidence , boxes, box_class_probs, threshold = 0.3):
    """
    通过阈值来过滤对象和分类的置信度。
    
    参数：
        box_confidence  - tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
        boxes - tensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。
        box_class_probs - tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。
        threshold - 实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。
    
    返回：
        scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。
        boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)
        classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引
        
    注意："None"是因为你不知道所选框的确切数量，因为它取决于阈值。
          比如：如果有10个锚框，scores的实际输出大小将是（10,）
    """
    
    #第一步：计算锚框的得分
    box_scores  = box_confidence * box_class_probs
    
    #第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)#（19*19*5*1）
    box_class_scores = K.max(box_scores, axis=-1)#找到最可能的类，是将最后一个维度进行展开（19*19*5*80）得到（19*19*5*1）
    
    #第三步：根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)
    
    #对scores, boxes 以及 classes使用掩码
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores , boxes , classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_eval(box_xy, box_wh, box_confidence, box_class_probs,image_shape, 
            max_boxes=15, score_threshold=0.2,iou_threshold=0.4):
    """
    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

    参数：
        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：
                        box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)
                        box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)
        image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)
        max_boxes - 整数，预测的锚框数量的最大值
        score_threshold - 实数，可能性阈值。
        iou_threshold - 实数，交并比阈值。
        
    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类
    """

    #获取YOLO模型的输出
    # image_input = Input(shape=(416,416, 3))
    # print("box_xy, box_wh :")
    # print(box_xy, box_wh)
    #中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy,box_wh)

    #可信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    #缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # #使用非最大值抑制
    # scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes




def pridect(origin_img,images,model):
    result=model.predict(images,batch_size=1)
    # print(result[0].shape)
    # print(result[1].shape)
    # print(result[2].shape)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    # num_layers = len(anchors)//3 # default setting
    # yolo_outputs = args[:num_layers]
    # y_true = args[num_layers:]
    # input_shape = K.cast(K.shape( result[0])[1:3] * 32, K.dtype(y_true[0]))
    # print(K.shape(result[0])[1:3] * 32)
    input_shape =K.shape(result[0])[1:3] * 32
    colors = yolo_utils.generate_colors(class_names)

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(result[0], anchors[anchor_mask[0]], 20,input_shape)
    scores, boxes, classes=yolo_eval(box_xy, box_wh, box_confidence, box_class_probs,image_shape=(float(origin_img.size[1]),float(origin_img.size[0])))
    for i in  range(1,3):
        box_xy, box_wh, box_confidence, box_class_probs = yolo_head(result[i], anchors[anchor_mask[i]], 20,input_shape)
        tmp_scores, tmp_boxes, tmp_classes=yolo_eval(box_xy, box_wh, box_confidence, box_class_probs,image_shape=(float(origin_img.size[1]),float(origin_img.size[0])))
        scores=tf.concat([scores,tmp_scores],axis=0)
        boxes=tf.concat([boxes,tmp_boxes],axis=0)
        classes=tf.concat([ classes, tmp_classes],axis=0)
        # yolo_utils.draw_boxes(origin_img, scores, boxes, classes, class_names, colors)



    #使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, 15, 0.4)
    yolo_utils.draw_boxes(origin_img, scores, boxes, classes, class_names, colors)
    # print("scores.shape = " + str(scores.shape))
    # print("boxes.shape = " + str(boxes.shape))
    # print("classes.shape = " + str(classes.shape))

    return origin_img


def camera_test(yolo_model):
    # 0是代表摄像头编号，只有一个的话默认为0

    url = 'http://192.168.2.106:4747/video?640x480'
    capture =cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920) 
    capture.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('road_Test_output.MP4',-1,25, (1920,1080))
    # capture.set(cv.CAP_PROP_FPS,30)
    while (True):

        ref, frame = capture.read()
        # frame = frame[:,::-1,:]
        img = cv.resize(frame, (int(416), int(416)), interpolation=cv.INTER_NEAREST)
        #cv2 frame to image
        img = Image.fromarray(np.uint8(img))
        x = image.img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        dec_image = np.vstack([x])
        images=Image.fromarray(np.uint8(frame))
        #pridect
        res=pridect(images,dec_image,yolo_model)
        #frame = cv.resize(frame, (96),(96))
        display=np.array(res)
        cv.imshow("Yolo", display)
        # cv.imshow("Yolo", frame)
        out.write(display)
        # 等待30ms显示图像，若过程中按“Esc”退出
        c = cv.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break


# yolo = YOLO()
class_names = yolo_utils.read_classes(current_path+"/model_data/voc_classes.txt")
anchors = yolo_utils.read_anchors(current_path+"/model_data/yolo_anchors.txt")
num_classes = len(class_names)
num_anchors = len(anchors)
input_shape = (416,416)
image_input = Input(shape=(None, None, 3))
h, w = input_shape
model= yolo_body(image_input, num_anchors//3, num_classes)
model.load_weights("logs/ep055-loss18.931-val_loss20.760.h5", by_name=True, skip_mismatch=True)
model.summary()
model.compile(optimizer='Adam',
            loss={
            'yolo_loss': lambda y_true, y_pred: y_pred},
            metrics=['accuracy'])

img = image.load_img("c:\\Users\\I1661\\Desktop\\cat.jpg", target_size=(416,416))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
model.predict(images)

camera_test(model)

# while(True):
#     path=input()
#     if (path!='exit'):
#             img = image.load_img(path, target_size=(416,416))
#             x = image.img_to_array(img)/255.0
#             x = np.expand_dims(x, axis=0)
#             images = np.vstack([x])
#             origin_img = image.load_img(path)

#             pridect(origin_img,images,model)

#             # plt.imshow(yolo.detect_image(origin_img)) # 显示图片
#             # plt.axis('off') # 不显示坐标轴
#             # plt.show()
#     else:
#         break

# yolo_outputs = yolo_head(model.output, anchors, len(class_names))
