import os
import json
import torch
import csv

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Directory
current_path = Path(os.path.dirname(os.path.abspath(__file__))) # ../HRNET/utils 의 절대 경로
parent_path = current_path.parent                               # ../HRNet/ 의 절대경로
data_path = parent_path / 'data/openai'                         # ../HRNet/data/openai의 절대 경로
annotation_dir = data_path / 'annotation'
image_dir = data_path / 'image'

csv_filepath = annotation_dir / 'annotation.csv'

img_map = {}
for (root, directories, files) in os.walk(image_dir):
    for file in files:
        if '.jpg' in file:
            file_path = os.path.join(root, file)
            basename = os.path.basename(file)
            img_map[basename] = file_path
            
print('img file # : {}'.format(len(img_map)))

csv_list = []
for (root, directores, files) in os.walk(annotation_dir):
    for file in files:
        if '.json' in file:                                     # 오직 .json 확장자만 리턴한다.
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as json_file:
                line = []
                # Empty list
                
                json_data = json.load(json_file)
                json_data_categories = dict(json_data['categories'][0])
                # type:string =["Lip", "Hand", "Person", "Dance"]
                # type_id:int =["0", "1", "2", "3"] 바로 위의 type:string에 대응되는 정수값
                # skeleton:list, int = 관절간 연결 정보 표현
                # keypoints : list , string = 관절 번호에 따라 각 관절 명 할당
                json_data_annotations = dict(json_data['annotations'][0])
                # id:int = 각 annotations와 매칭되는 이미지의 고유 번호
                # bbox:list, int = 바운딩 박스 영역의 Left-Top 위치 및 가로, 세로 크기
                # num_keypoints:int = 해당 영상에서 마킹된 관절 수
                # keypoints:list, int = 관절 번호 순서대로 (x, y, visible)값으로 위치를 표현한다.
                #                       visible :: 0=마킹되지 않은 관절, 1=마킹은 되었으나 보이지 않은 관절, 2=마킹되고 보이는 관절
                json_data_images = dict(json_data['images'][0])
                # file_name:str = basename of img
                
                line.append(json_data_annotations['id'])                # append 'id'
                line.append(json_data_images['file_name'])              # append 'file_name(basename)'
                line.append(img_map[json_data_images['file_name']])     # append img_path
                line.append(json_data_annotations['bbox'])              # append bounding box
                line.append(json_data_annotations['num_keypoints'])     # append num_keypoints
                line.append(json_data_annotations['keypoints'])         # append keypoints
                
                csv_list.append(line)

index = ['id', 'file_name', 'file_path', 'bbox', 'num_keypoints', 'keypoints']
with open(csv_filepath, 'w', encoding='utf-8', newline='') as cf:
    wr = csv.writer(cf)
    wr.writerow(index)
    for l in csv_list:
        wr.writerow(l)
