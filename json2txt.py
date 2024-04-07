import os
import json
import math

folder_path = './parking_set/labels/'

output_file = './parking_set/five_freedom_labels/'

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(folder_path, filename)

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        output_txt_file = output_file + data['metadata']['file_name'].split('.')[0] + ".txt"

        if len(data['objects']) != 0:
            # save to txt fold 4->1->2->3
            with open(output_txt_file, 'w') as txt_file:
                for i in range(len(data['objects'])):
                    point1 = data['objects'][i]['point_list'][0]
                    point2 = data['objects'][i]['point_list'][1]
                    point3 = data['objects'][i]['point_list'][2]
                    point4 = data['objects'][i]['point_list'][3]
                    length12 = math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
                    length34 = math.sqrt((point4[0]-point3[0])**2+(point4[1]-point3[1])**2)
                    maxlength = max(length12, length34)
                    parking_points = point4 + point1
                    parking_points.append(maxlength)
                    parking_points_string = ' '.join(map(str, parking_points))
                    txt_file.write(f'0 {parking_points_string}\n')
        else:
            with open(output_txt_file, 'w') as txt_file:
                pass

        print(f'内容已保存到{output_txt_file}文件中。')
