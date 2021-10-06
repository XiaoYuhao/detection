import torch
import torchvision
from torchvision.models.detection.roi_heads import fastrcnn_loss
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO数据集标签对照表
COCO_CLASSES ={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
            '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
            '#ffd8b1', '#e6beff', '#808080']

# 为每一个标签对应一种颜色，方便我们显示
LABEL_COLOR_MAP = {k: COLORS[i%len(COLORS)] for i, k in enumerate(COCO_CLASSES.keys())}

# 判断GPU设备是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def faster_rcnn_detection(img_path):
    # 加载pytorch自带的预训练Faster RCNN目标检测模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    # 读取输入图像，并转化为tensor
    origin_img = Image.open(img_path, mode='r').convert('RGB')
    img = TF.to_tensor(origin_img)
    img = img.to(device)

    # 将图像输入神经网络模型中，得到输出
    output = model(img.unsqueeze(0))

    labels = output[0]['labels'].cpu().detach().numpy()     # 预测每一个obj的标签
    scores = output[0]['scores'].cpu().detach().numpy()     # 预测每一个obj的得分
    bboxes = output[0]['boxes'].cpu().detach().numpy()      # 预测每一个obj的边框
    # 这个我们只选取得分大于0.8的
    obj_index = np.argwhere(scores>0.8).squeeze(axis=1).tolist()

    # 使用ImageDraw将检测到的边框和类别打印在图片中，得到最终的输出
    draw = ImageDraw.Draw(origin_img)
    font = ImageFont.truetype('Arial.ttf', 15)

    for i in obj_index:
        box_location = bboxes[i].tolist()
        draw.rectangle(xy=box_location, outline=LABEL_COLOR_MAP[labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=LABEL_COLOR_MAP[labels[i]])

        text_size = font.getsize(COCO_CLASSES[labels[i]])
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=LABEL_COLOR_MAP[labels[i]])
        draw.text(xy=text_location, text=COCO_CLASSES[labels[i]], fill='white', font=font)

    del draw

    origin_img.save("result.png")


if __name__ == '__main__':
    faster_rcnn_detection("input.jpg")


