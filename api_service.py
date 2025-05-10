
import os
import random
import re
import sys

import cv2
from flask import send_file
import json, base64
from io import BytesIO
from PIL import Image
import io
import numpy as np
from datetime import datetime, timedelta
from datetime import date
from utility import is_color_code_valid, is_valid_scale_params, is_valid_position_params, upload_source_image_file
from model_test import do_mask

def service_remove_background(request):
    data = {}
    source_image_pil = None
    bg_image_pil = None
    output_file_format = 'png'
    client = 'api'
    scale = 0
    position = 'center'

    if 'source_image_file' not in request.files and 'source_image_url' not in request.form and 'source_image_base64' not in request.form:
        data['error'] = "source image not found"
        data['error_code'] = 1003
        return json.dumps(data), 400
    elif 'source_image_file' in request.files:
        try:
            source_image = request.files.get('source_image_file')
            source_image_pil = Image.open(source_image);
            if source_image_pil.format == 'GIF':
                data['error'] = "invalid source image file"
                data['error_code'] = 1004
                return json.dumps(data), 400
            source_image_file_size = len(source_image.read())
            source_image_file_size_mb = source_image_file_size / 1000000
            if source_image_file_size_mb > 12:
                data['error'] = "source image size large than 12mb"
                data['error_code'] = 1005
                return json.dumps(data), 400
        except  Exception as e:
            data['error'] = "invalid source image file"
            data['error_code'] = 1004
            return json.dumps(data), 400
    elif 'source_image_url' in request.form:
        source_image_url = request.form['source_image_url']
        source_image_pil, source_image_file_size_mb = is_url_image(source_image_url)
        if source_image_pil is None:
            data['error'] = "invalid source image url"
            data['error_code'] = 1006
            return json.dumps(data), 400
        if source_image_file_size_mb > 12:
            data['error'] = "source image size large than 12mb"
            data['error_code'] = 1005
            return json.dumps(data), 400
    elif 'source_image_base64' in request.form:
        source_image_base64 = request.form['source_image_base64']
        source_image_base64 = re.sub(r"data:image/(png|jpg|jpeg);base64,", '', source_image_base64)
        try:
            imgdata = base64.b64decode(source_image_base64)
            source_image = BytesIO(imgdata)
            source_image_pil = Image.open(source_image)
        except Exception as e:
            data['error'] = "invalid base64 string."
            data['error_code'] = 1007
            return json.dumps(data), 400

        source_image_file_size = len(source_image.read())
        source_image_file_size_mb = source_image_file_size / 1000000
        if source_image_file_size_mb > 12:
            data['error'] = "source image size large than 12mb"
            data['error_code'] = 1005
            return json.dumps(data), 400
    if 'format' in request.form:
        output_file_format = request.form['format']
        if output_file_format not in ('png', 'jpg'):
            data['error'] = "wrong output format"
            data['error_code'] = 1008
            return json.dumps(data), 400

    if 'bg_color_code' in request.form:
        bg_color_code = request.form['bg_color_code']
        if not is_color_code_valid(bg_color_code):
            data['error'] = "invalid background color code format"
            data['error_code'] = 1009
            return json.dumps(data), 400
    if 'bg_image_url' in request.form:
        bg_image_url = request.form['bg_image_url']
        bg_image_pil, bg_image_file_size_mb = is_url_image(bg_image_url)
        if bg_image_pil is None:
            data['error'] = "invalid background image url"
            data['error_code'] = 1010
            return json.dumps(data), 400
        if bg_image_file_size_mb > 12:
            data['error'] = "background image size large than 12mb"
            data['error_code'] = 1011
            return json.dumps(data), 400
    if 'bg_image_base64' in request.form:
        bg_image_base64 = request.form['bg_image_base64']
        bg_image_base64 = re.sub(r"data:image/(png|jpg|jpeg);base64,", '', bg_image_base64)
        try:
            img_data = base64.b64decode(bg_image_base64)
            bg_image = BytesIO(img_data)
            bg_image_pil = Image.open(bg_image)
        except Exception as e:
            data['error'] = "invalid background image base64 string."
            data['error_code'] = 1012
            return json.dumps(data), 400
    if 'bg_image_file' in request.files:
        try:
            bg_image = request.files.get('bg_image_file')
            bg_image_pil = Image.open(bg_image)
            bg_image_file_size = len(bg_image.read())
            bg_image_file_size_mb = bg_image_file_size / 1000000
            if bg_image_file_size_mb > 12:
                data['error'] = "background image size large than 12mb"
                data['error_code'] = 1011
                return json.dumps(data), 400
        except Exception as e:
            data['error'] = "invalid background image format"
            data['error_code'] = 1013
            return json.dumps(data), 400

    if 'frequenteraug' in request.form:
        client = request.form['frequenteraug']

    if 'client' in request.form:
        client = request.form['client']

    if 'scale' in request.form:
        scale = request.form['scale'].strip()
        if not is_valid_scale_params(scale):
            data['error'] = "invalid scale parameter"
            data['error_code'] = 1016
            return json.dumps(data), 400
        scale = int(scale.replace('%', ''))

    if 'scale' in request.form and 'position' in request.form:
        position = request.form['position'].strip()
        if not is_valid_position_params(position):
            data['error'] = "invalid position parameter"
            data['error_code'] = 1017
            return json.dumps(data), 400

 
    icc_profile = source_image_pil.info.get(
        "icc_profile") if 'icc_profile' in source_image_pil.info else None
    source_image_pil = upload_source_image_file(source_image_pil, output_file_format, client)
    full_size_output_image = do_mask(source_image_pil, request, bg_image_pil, scale, position)

   
    np_arr = np.array(full_size_output_image)
    # print(channel)
    channel=''
    if channel == 'alpha':
        np_arr[np_arr < 30] = 0
        np_arr[np_arr > 230] = 255
    elif channel == 'original_alpha':
        pass
        # np_arr[np_arr < 30] = 0
        # np_arr[np_arr > 230] = 255
    else:
        # print(full_size_output_image.mode)
        if full_size_output_image.mode == 'RGBA':
            # pass
            # alpha_channel = np_arr[:, :, 3]
            # alpha_channel[alpha_channel < 15] = 0
            # alpha_channel[alpha_channel > 240] = 255
            # np_arr[:, :, 3] = alpha_channel
            np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGRA2RGBA)
        else:
            np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB)
            if 'format' not in request.form:
                output_file_format = 'jpg'
    if icc_profile is None or channel == 'alpha' or channel == 'original_alpha':
        is_success, im_buf_arr = cv2.imencode("." + output_file_format, np_arr, )
        byte_im = im_buf_arr.tobytes()
        return send_file(io.BytesIO(byte_im),
                         mimetype='image/png' if output_file_format == 'png' else 'image/jpeg'), 200
    else:
        # if output_file_format == 'png':
        #     np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGRA2RGBA)
        # else:
        #     np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGRA2RGB)
        # img = Image.fromarray(np_arr)
        img = full_size_output_image
        buf = io.BytesIO()
        img.save(buf, format='PNG' if output_file_format == 'png' else 'JPEG', icc_profile=icc_profile)
        byte_im = buf.getvalue()
        return send_file(io.BytesIO(byte_im),
                             mimetype='image/png' if output_file_format == 'png' else 'image/jpeg'), 200