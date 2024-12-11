import torchvision
from flask import Flask, jsonify, request
from PIL import Image
import os
import numpy as np
import openslide
import torch


# flask
app = Flask(__name__)


@app.route('/getPic/', methods=['POST'])
def getPic():
    # preprocess

    slide_path = request.json.get('image_path')

    #print(files_list[i][:7])
    slide = openslide.OpenSlide(slide_path)


    save_path = os.path.dirname(slide_path)
    file_name = os.path.splitext(os.path.basename(slide_path))[0]

    for j in range(1,6,1):
        downsample = int(40/int(slide.level_downsamples[j]))
        #print(j,downsample)

        now_file_path = os.path.join(save_path, file_name)
        #print(now_file_path)


        now_file_path = os.path.join(now_file_path, str(downsample))
        #print(now_file_path)


        if not os.path.exists(now_file_path):
            os.makedirs(now_file_path)

        # size
        width, height = slide.level_dimensions[j]
        #print(width, height)

        for x in range(0, width, 512):
            #print(x)
            x_folder = os.path.join(now_file_path, str(int(x/512)))
            #print(x_folder)
            os.makedirs(x_folder, exist_ok=True)

            for y in range(0, height, 512):

                #print(x,y)
                img = slide.read_region((int(x)*int(slide.level_downsamples[j]), int(y)*int(slide.level_downsamples[j])), j, (512, 512)).convert('RGB')
                patch_name = os.path.join(x_folder, str(int(y/512))+'.png')

                #print(patch_name)

                img.save(patch_name)

    return jsonify({'message': 'Image processed and saved', 'saved_path': save_path}), 0
    # return 0



if __name__ == '__main__':
    app.run()

