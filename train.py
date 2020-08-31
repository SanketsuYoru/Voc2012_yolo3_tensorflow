#utils
import os
import random
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt # plt 用于显示图片
# TensorFlow and tf.keras
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers import Adam
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



def yolo_filter_boxes(box_confidence , boxes, box_class_probs, threshold = 0.6):
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

#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    #特征层数
    num_layers = len(anchors)//3
    #特征层遮罩(这里不考虑yolo-tiny的情况所以特征层遮罩有三组)
    #---------------------------------------------------#
    #  [6,7,8]->52x52
    #  [3,4,5]->26x26
    #  [0,1,2]->13x13
    #---------------------------------------------------#
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32') # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:]
    true_boxes[..., 2:4] = boxes_wh/input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true



def yolo_eval(box_xy, box_wh, box_confidence, box_class_probs,image_shape=(2500.,2500.), 
            max_boxes=10, score_threshold=0.5,iou_threshold=0.5):
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
    print("box_xy, box_wh :")
    print(box_xy, box_wh)
    #中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy,box_wh)

    #可信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    #缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    #使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def pridect(origin_img,images,model):
    result=model.predict(images,batch_size=1)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    # num_layers = len(anchors)//3 # default setting
    # yolo_outputs = args[:num_layers]
    # y_true = args[num_layers:]
    # input_shape = K.cast(K.shape( result[0])[1:3] * 32, K.dtype(y_true[0]))
    print(K.shape(result[0])[1:3] * 32)
    input_shape =K.shape(result[0])[1:3] * 32
    colors = yolo_utils.generate_colors(class_names)
    for i in  range(0,3):
        box_xy, box_wh, box_confidence, box_class_probs = yolo_head(result[i], anchors[anchor_mask[i]], 20,input_shape)
        scores, boxes, classes=yolo_eval(box_xy, box_wh, box_confidence, box_class_probs,image_shape=(float(origin_img.size[1]),float(origin_img.size[0])))
        print(scores)
        print(boxes)
        print(classes)
        yolo_utils.draw_boxes(origin_img, scores, boxes, classes, class_names, colors)
    # print("scores.shape = " + str(scores.shape))
    # print("boxes.shape = " + str(boxes.shape))
    # print("classes.shape = " + str(classes.shape))

    plt.imshow(origin_img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()






log_dir = 'logs/'
annotation_path = '2012_train.txt'
# 类和anchorsBox的数量
class_names = yolo_utils.read_classes(current_path+"/model_data/voc_classes.txt")
anchors = yolo_utils.read_anchors(current_path+"/model_data/yolo_anchors.txt")
num_classes = len(class_names)
num_anchors = len(anchors)
input_shape = (416,416)
image_input = Input(shape=(None, None, 3))
h, w = input_shape
model= yolo_body(image_input, num_anchors//3, num_classes)
model.load_weights("logs/ep051-loss18.915-val_loss21.332.h5", by_name=True, skip_mismatch=True)
# model = load_model("model_data/yolo.h5")


# y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
#     num_anchors//3, num_classes+5)) for l in range(3)]
# loss_input = [*model.output, *y_true]
# model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
#     arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(loss_input)



# 训练参数设置
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)



batch_size = 10
# 0.1用于验证，0.9用于训练
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val



# freeze_num = 185
# for i in range(freeze_num): model.layers[i].trainable = False
# print('\nFreeze the first {} layers of total {} layers.\n'.format(freeze_num, len(model.layers)))

# print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

# #compile the model with custom loss function 
y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
    num_anchors//3, num_classes+5)) for l in range(3)]

model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
    [*model.output, *y_true])
model = Model([model.input, *y_true], model_loss)
model.summary()
# model.compile(optimizer=Adam(lr=1e-3), loss={
#     # use custom yolo_loss Lambda layer.
#     'yolo_loss': lambda y_true, y_pred: y_pred})


# # print('lines.length {}  batch_size {} input_shape {} anchors{}num_classes{}.'.format(len(lines), batch_size,str(input_shape),anchors.shape,num_classes))
# model.fit(
#     data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#         steps_per_epoch=max(1, num_train//batch_size),
#         validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#         validation_steps=max(1, num_val//batch_size),
#         epochs=50,
#         initial_epoch=0,
#         callbacks=[logging, checkpoint])
# model.save_weights(log_dir + 'trained_weights_stage_1.h5')


#Adjust batch size to 3 to prevent run out of memory
batch_size = 3
print('\nUnfreeze all layers.\n')
for i in range(len(model.layers)):
    model.layers[i].trainable = True
#Compile to apply changes
model.compile(optimizer=Adam(lr=1e-7), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

history=model.fit(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    steps_per_epoch=max(1, num_train//batch_size),
    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    validation_steps=max(1, num_val//batch_size),
    epochs=100,
    initial_epoch=50,
    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
model.save_weights(log_dir + 'trained_weights_final.h5')


# #graph
# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(loss)) # Get number of epochs

# #------------------------------------------------
# # Plot training and validation accuracy per epoch
# #------------------------------------------------
fig=plt.figure('Training and validation')  
# sub_1 = fig.add_subplot(1,2,1)
# sub_1.plot(epochs, acc,'y',label='training')
# sub_1.plot(epochs, val_acc, 'b',label='validation')
# sub_1.set_title('Accuracy')
# sub_1.legend()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
sub_2 = fig.add_subplot(1,2,2)
sub_2.plot(epochs, loss,'y',label='training')
sub_2.plot(epochs, val_loss, 'b',label='validation')
sub_2.set_title('Loss')
sub_2.legend() 
plt.show()




# print('please input image path , input exit to finish')

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
