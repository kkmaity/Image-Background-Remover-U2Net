import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset1
from model import NOTBGNET
import cv2
from skimage import io, transform
from matting import apply_mask_trimap


Image.MAX_IMAGE_PIXELS = None
import base64
from io import BytesIO
import os, sys

model_name='u2net'#u2netp
# source_image_path = os.path.join(os.getcwd(), 'test_data', 'test_images/s3.jpg')
# bg_image_path = os.path.join(os.getcwd(), 'test_data', 'test_images/bg.jpg')
# output_image_path = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
model_dir = os.path.join(os.getcwd(), 'ai_models', model_name, model_name + '.pth')

net = NOTBGNET(3, 1)

net.load_state_dict(torch.load(model_dir, map_location='cpu'))

if torch.cuda.is_available():
    print("CUDA AVAILABLE")
    net.cuda()
else:
    print("CUDA NOT AVAILABLE")
net.eval()


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn



def get_img_bbox(img):
    np_img = np.array(img)
    img_ch3 = np_img[:, :, 3]
    img_ch3[img_ch3 < 20] = 0
    np_img[:, :, 3] = img_ch3
    im = Image.fromarray(np_img)
    return im.getbbox()


def trim(img, border=0):
    width, height = img.size
    bbox = get_img_bbox(img)
    if bbox:
        left, upper, right, lower = bbox
        bbox = (max((left - border), 0), max((upper - border), 0), min((right + border), width),
                min((lower + border), height))
        return img.crop(bbox)
    return img


def generate_full_size_output(output, request, bg_image_pil, scale, position):
    width, height = output.size

    if 'roi' in request.form:
        roi = request.form['roi']
        if 'px' in roi:
            roi = roi.replace('px', '')
            val = roi.split(',')
            x1, y1, x2, y2 = int(val[0]), int(val[1]), int(val[2]), int(val[3])
        else:
            roi = roi.replace('%', '')
            val = roi.split(',')
            x1, y1, x2, y2 = int(width * int(val[0]) / 100), int(height * int(val[1]) / 100), int(
                width * int(val[2]) / 100), int(height * int(val[3]) / 100)

        x1 = 0 if x1 >= width else x1
        y1 = 0 if y1 >= height else y1
        x2 = width if x2 >= width else x2
        y2 = height if y2 >= height else y2
        output = output.crop((x1, y1, x2, y2))
        width, height = output.size

    
    if 'scale' in request.form:
        scale_output = trim(output, 0)
        scale_output_width, scale_output_height = scale_output.size
        canvas_output_width, canvas_output_height = int(width * int(scale) / 100), int(height * int(scale) / 100)
        _w, _h = 0, 0
        if scale_output_width > scale_output_height:
            _w, _h = canvas_output_width, int(canvas_output_width / scale_output_width * scale_output_height)
            if _h > canvas_output_height:
                _w, _h = int(canvas_output_height / scale_output_height * scale_output_width), canvas_output_height
        else:
            _w, _h = int(canvas_output_height / scale_output_height * scale_output_width), canvas_output_height
            if _w > canvas_output_width:
                _w, _h = canvas_output_width, int(canvas_output_width / scale_output_width * scale_output_height)

        resize_scale_output = scale_output.resize((_w, _h), Image.ANTIALIAS)
        resize_scale_output_w, resize_scale_output_h = resize_scale_output.size
        img = Image.new('RGBA', (width, height), (255, 0, 0, 0))
        if position == 'center':
            ww, hh = int((width - _w) / 2), int((height - _h) / 2)
        elif ',' in position:
            position = position.replace('%', '')
            p0, p1 = int(position.split(',')[0]), int(position.split(',')[1])
            ww = int(width * p0 / 100)
            hh = int(height * p1 / 100)
            if ww + resize_scale_output_w > width:
                ww = width - resize_scale_output_w
            if hh + resize_scale_output_h > height:
                hh = height - resize_scale_output_h
        else:
            position = position.replace('%', '')
            p0, p1 = int(position), int(position)
            ww = int(width * p0 / 100)
            hh = int(height * p1 / 100)
            if ww + resize_scale_output_w > width:
                ww = width - resize_scale_output_w
            if hh + resize_scale_output_h > height:
                hh = height - resize_scale_output_h

        img.paste(resize_scale_output, (ww, hh))
        output = img

    if 'bg_color_code' in request.form:
        background = Image.new("RGB", output.size, request.form['bg_color_code'])
        # background.show()
        np_arr = np.array(output)
        alpha_channel = np_arr[:, :, 3]
        # alpha_channel[alpha_channel < 20] = 0
        # alpha_channel[alpha_channel > 235] = 255
        np_arr[:, :, 3] = alpha_channel
        # np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGRA2RGBA)
        output = Image.fromarray(np_arr)
        background.paste(output, mask=output.split()[3])
        return background
    elif bg_image_pil is not None:
        background = bg_image_pil
        w , h = bg_image_pil.size
        bg_width , bg_height = 0, 0
        if width > height :
            bg_width = width
            bg_height = int(h* bg_width/w)
            if height > bg_height:
                bg_height = height
                bg_width = int(w * bg_height / h)
        else:
            bg_height = height
            bg_width = int(w*bg_height/h)
            if width > bg_width:
                bg_width = width
                bg_height = int(h * bg_width / w)

        background = background.resize((bg_width,bg_height), Image.Resampling.LANCZOS)

        if bg_width == width:
            x1 = 0
            y1  = int ((bg_height -  height ) /2 )
            x2 = x1 + width
            y2 = y1 + height
        else:
            x1 = int ((bg_width -  width ) /2 )
            y1 = 0
            x2 = x1 + width
            y2 = y1 + height
        background = background.crop((x1, y1, x2, y2))
        np_arr = np.array(output)
        alpha_channel = np_arr[:, :, 3]
        alpha_channel[alpha_channel < 20] = 0
        alpha_channel[alpha_channel > 235] = 255
        np_arr[:, :, 3] = alpha_channel
        # np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGRA2RGBA)
        output = Image.fromarray(np_arr)
        background.paste(output, mask=output.split()[3])
        return background
    else:
        return output







def save_output(source_image,pred,request,bg_image_pil,scale,position):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = predict_np * 255
    orig = source_image.convert('RGB')
    output, mask = apply_mask_trimap(orig, predict_np)

    # im = Image.fromarray(predict_np).convert('RGB')
    # img_name = source_image.split(os.sep)[-1]
    # image = io.imread(source_image)
    # imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    # pb_np = np.array(imo)
    # masks = pb_np[:,:,0]
    # masks = np.expand_dims(masks, axis=2)
    # imo = np.concatenate((image, masks),axis =2)
    # imo = Image.fromarray(imo, 'RGBA')

    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]

    # imo.save(output_image+imidx+'.png')
    
    return generate_full_size_output(output, request, bg_image_pil, scale, position)



def do_mask(source_image_pil,request,bg_image_pil, scale, position):
    # try:
        # print(source_image_path)
        # source_image_pil = Image.open(source_image_path)
        # bg_image_pil = Image.open(bg_image_path)

        source_image_pil_resize = source_image_pil
        width, height = source_image_pil.size
        sz = 600
        if width > sz or height > sz:
            if width > height:
                width = sz
                height = int(height * sz / width)
            else:
                width = int(width * sz / height)
                height = sz
            source_image_pil_resize = source_image_pil_resize.resize((width, height), Image.Resampling.LANCZOS)

        img_name_list = [source_image_pil_resize]
        test_salobj_dataset = SalObjDataset1(img_name_list=img_name_list,
                                            lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )


        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader,0):

            with torch.no_grad():
                inputs_test = data_test['image']
                inputs_test = inputs_test.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)

                d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

                # normalization
                pred = d1[:, 0, :, :]
                pred = normPRED(pred)
               # save results to test_results folder
                res = save_output(source_image=source_image_pil,pred=pred,request=request,bg_image_pil=bg_image_pil,scale=scale,position=position)
                
                del d1, d2, d3, d4, d5, d6, d7, pred, inputs_test
                torch.cuda.empty_cache()
                return res

