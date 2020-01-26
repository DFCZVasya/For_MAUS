import cv2
from PIL import Image

def take_and_resize(frame, output_size = 314):
    crop_coords = []
    height = frame.shape[0]
    width = frame.shape[1]
    min_side = min(height, width)
    frame = Image.fromarray(frame)
    if min_side == height:
        crop_coords.append(int((width / 2) - (height /2)))
        crop_coords.append(int((width / 2) + (height /2)))
        frame = frame.crop((crop_coords[0], 0, crop_coords[1], height))
        frame = frame.resize((output_size,output_size),Image.ANTIALIAS)
    else:
        pass
    
    return frame
