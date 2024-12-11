import torchvision
from flask import Flask, jsonify, request
from PIL import Image
import os
import numpy as np

from options.test_options import TestOptions
from models import create_model
# os.add_dll_directory(r'D:\openslide\openslide-win64-20230414\openslide-win64-20230414\bin')
import openslide
import torch

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

# flask
app = Flask(__name__)

# read model, It is better to treat the model as a global variable

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)  # create a model given opt.model and other options
# print(model)

model.save_dir = "./F2HE_512_fast"
model.load_networks(5)

@app.route('/imageHD/', methods=['POST'])
def imageHD():
    # preprocess

    image_path = request.json.get('image_path')

    device = torch.device("cuda")


    if not image_path or not os.path.isfile(image_path):
        return jsonify({'error': 'Invalid image path'}), 400000

    '''
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image)).convert('RGB')
    '''


    # create save dir
    SAVE_DIR = os.path.dirname(image_path)
    SAVE_DIR = os.path.join(SAVE_DIR,"hd")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # preprocess
    image = Image.open(image_path).convert('RGB')

    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

    image = image.to(device)
    image = torch.unsqueeze(image, 0)

    # prediction
    model.netG.eval()
    with torch.no_grad():
        output = model.netG(image)
    # save img

    image_pil = tensor2im(output)

    base_name = os.path.basename(image_path)
    save_path = os.path.join(SAVE_DIR, base_name)

    save_image(image_pil, save_path)

    return jsonify({'message': 'Image processed and saved', 'saved_path': save_path}), 0
    # return 0

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

