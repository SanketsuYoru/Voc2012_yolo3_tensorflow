import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
current_path = os.path.dirname(__file__)
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open(current_path+'/VOC2012/Annotations/%s.xml'%(image_id))
    out_file = open(current_path+'/VOC2012/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists(current_path+'/VOC2012/labels/'):
        os.makedirs(current_path+'/VOC2012/labels/')
    image_ids = open(current_path+'/VOC2012/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open(current_path+'/2012_%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write(current_path+'/VOC2012/JPEGImages/%s.jpg\n'%(image_id))
        convert_annotation(year, image_id)
    list_file.close()

