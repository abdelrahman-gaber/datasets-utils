import os
import json
import numpy as np
import argparse
from scandir import scandir
import cv2
import sys

sys.path.append("../PyramidBox-Pytorch/")
import detection_fns
from detection_fns import *

images_input_folder = "/media/sdf/COCO/2014/images/val2014"
json_input_folder = "/media/sdf/COCO/2014/Annotations/minival2014"
output_folder = "/media/sdf/COCO/2014/Annotations_person"

for files in scandir(json_input_folder):
    person_exists = False
    if files.is_file() and files.name.endswith('.json'):
        json_file = os.path.join(json_input_folder, files.name)
        print(json_file)
        out_file = output_folder + "/" + os.path.splitext(files.name)[0] + ".txt"
        print(out_file)
        image_path = images_input_folder + "/" + os.path.splitext(files.name)[0] + ".jpg";
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #f = open(out_file, 'w')
        annos = json.load(open(json_file, 'r'))
        anno = annos["annotation"]
        if not anno:
            continue
        #print(len(anno))
        for obj in anno:
            category_id = obj["category_id"]
            is_crowd = obj["iscrowd"]
            if category_id == 1 and is_crowd != 1:
                person_exists = True
                f = open(out_file, 'a')
                bbox = obj["bbox"]
                #print(bbox)
                P_xmin = bbox[0]
                P_ymin = bbox[1] 
                P_xmax = bbox[2] + bbox[0]
                P_ymax = bbox[3] + bbox[1] 
                person_bbox = np.atleast_2d( np.asarray([1, P_xmin, P_ymin, P_xmax, P_ymax]) ) # [CLASS = 1, DETS]
                np.savetxt(f, person_bbox, fmt=["%d",]*5 , delimiter=" ")
                cv2.rectangle(image, (int(P_xmin),int(P_ymin)), (int(P_xmax),int(P_ymax)), (0, 255, 0), 1)
    
    if person_exists == True:
        cv2.imshow("img", image)
        cv2.waitKey(0)

