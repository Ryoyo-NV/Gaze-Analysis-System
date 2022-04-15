import PIL.Image
import numpy as np
import cv2
#from torchvision.transforms.functional import crop


def square_box(bbox, scale=1.0):
    
    # get box shape
    x0, y0, x1, y1 = bbox[0:4]
    w, h = x1 - x0, y1 - y0
    # scale square box
    new_w = new_h = max(w, h) * scale
    
    # shift origin pixels
    new_x0 = x0 - (new_w - w) / 2.0
    new_y0 = y0 - (new_h - h) / 2.0
    
    return (new_x0, new_y0, new_x0 + new_w, new_y0 + new_h)
    
    
#def crop_box(image, bbox):
#    
#    x0, y0, x1, y1 = bbox[0:4]
#    h = max(1, y1 - y0)
#    w = max(1, x1 - x0)
#    
#    return crop(image, y0, x0, h, w)


class CropResize(object):

    def __init__(self, shape, square=False, scale=1.0, channels=False):
        self.shape = shape
        self.square = square
        self.scale = scale
        self.channels = channels

    def __call__(self, image, boxes):
        num_box = len(boxes)
        crops_shape = []
        if self.channels:
          crops_shape = (num_box, self.shape[1], self.shape[0], 3)
        else:
          crops_shape = (num_box, self.shape[1], self.shape[0])

        crops = np.empty(crops_shape, dtype=image.dtype)
        new_boxes = boxes[:, :4].copy()
        image = PIL.Image.fromarray(image)

        for i, box in enumerate(boxes):

            box = np.array(square_box(box, self.scale))
            new_boxes[i] = box
            box_crop = np.array(crop_box(image, box))
            if self.channels:
                crops[i, :, :, :] = cv2.resize(box_crop, self.shape)
            else:
                crops[i, :, :] = cv2.resize(box_crop, self.shape)

        return crops, new_boxes


class BoxDraw(object):

    def __init__(self, thickness=2):
        self.thickness = thickness

    def __call__(self, image, boxes, colors=(0, 255, 0)):

        for i, b in enumerate(boxes):

            # get box color
            if isinstance(colors, tuple):
                color = colors
            elif isinstance(colors, list):
                color = colors[i]

            x0, y0 = int(b[0]), int(b[1])
            x1, y1 = int(b[2]), int(b[3])

            cv2.rectangle(image, (x0, y0), (x1, y1), color, self.thickness)

class TextDraw(object):

    def __init__(self, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 255, 255)):
        self.thickness = thickness
        self.color = color
        self.font = font
        self.font_scale = font_scale

    def __call__(self, image, faces):
        
        for i, face in enumerate(faces):

            x0, y0 = int(face["bbox"]["left"]), int(face["bbox"]["top"])
            x1, y1 = int(face["bbox"]["right"]), int(face["bbox"]["bottom"])

            age = "{}".format(int(face["age"]))
            
            if face["gender"] >= 0.6:
                gender = 'M'
            else:
                gender = 'F'

            label = "{}, {}".format(age, gender)

            size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0] 
            cv2.rectangle(image, (x0, y0 - size[1]), (x0 + size[0], y0), (100, 100, 100), cv2.FILLED)
            cv2.putText(image, label, (x0, y0), self.font, self.font_scale, self.color, self.thickness, lineType=cv2.LINE_AA)



