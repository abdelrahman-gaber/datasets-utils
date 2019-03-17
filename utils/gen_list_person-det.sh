#COCO train
python2.7 generate-lists.py --images_dir /media/sdf/COCO/2014/images/train2014 --annot_dir /media/sdf/person-detection/COCO/train --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode train --seperator_images COCO  --seperator_annot person-detection

# COCO val
python2.7 generate-lists.py --images_dir /media/sdf/COCO/2014/images/val2014 --annot_dir /media/sdf/person-detection/COCO/val --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode val --seperator_images COCO --seperator_annot person-detection

# AI-Challenger train 
python2.7 generate-lists.py --images_dir /media/sdf/AI-Challenger/original/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103 --annot_dir /media/sdf/person-detection/AI-Challenger/ai_challenger_keypoint_test_a_20180103 --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode train --seperator_images AI-Challenger --seperator_annot person-detection

python2.7 generate-lists.py --images_dir /media/sdf/AI-Challenger/original/ai_challenger_keypoint_test_b_20180103/keypoint_test_b_images_20180103 --annot_dir /media/sdf/person-detection/AI-Challenger/ai_challenger_keypoint_test_b_20180103 --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode train --seperator_images AI-Challenger --seperator_annot person-detection

python2.7 generate-lists.py --images_dir /media/sdf/AI-Challenger/original/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911 --annot_dir /media/sdf/person-detection/AI-Challenger/ai_challenger_keypoint_validation_20170911 --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode train --seperator_images AI-Challenger --seperator_annot person-detection

# CrowdHuman train
python2.7 generate-lists.py --images_dir /media/sdf/CrowdHuman/train/Images --annot_dir /media/sdf/person-detection/CrowdHuman/train --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode train --seperator_images CrowdHuman --seperator_annot person-detection

# CrowdHuman val
python2.7 generate-lists.py --images_dir /media/sdf/CrowdHuman/val/Images --annot_dir /media/sdf/person-detection/CrowdHuman/val --out_directory /home/ubuntu/caffe-NVIDIA/data/person-detection --mode val --seperator_images CrowdHuman --seperator_annot person-detection


