import re, math
import os, os.path
from datetime import datetime, timedelta
from datetime import date
import string
import uuid
import time
import os, os.path
from PIL import Image
import requests
from io import BytesIO
import gdown
from PIL import ExifTags
import sys

regex_hex_color_code = r"(^#(?:[0-9a-fA-F]{3}){1,2}$)"    
scale_params_regx = r"^([1-9][0-9]|100)%$"
position_params_regex = r"^center$|^([0-9]|[1-9][0-9]|100)%$|^([0-9]|[1-9][0-9]|100)%,([0-9]|[1-9][0-9]|100)%$"

IMAGE_DOWNLOAD_FOLDER_PATH = 'static/downloads'
IMAGE_DOWNLOAD_FOLDER_PATH_TEMP = 'static/download_temp'



def is_color_code_valid(color_code):
    if re.search(regex_hex_color_code, color_code):
        return True
    else:
        return False
    
def is_valid_scale_params(scale_params):
    if re.search(scale_params_regx, scale_params):
        return True
    else:
        return False


def is_valid_position_params(position_params):
    if re.search(position_params_regex, position_params):
        return True
    else:
        return False
    
    
def upload_source_image_file(source_image_pil, output_file_format, client, preview=False):
    try:
        flag = False
        width, height = source_image_pil.size
        if not preview:
            img_mp_10 = 10
            img_mp_25 = 25
            if output_file_format == 'png' and width * height > img_mp_10 * 1000000 and client != 'windows' and client != 'mac':
                flag = True
                width, height = resize_megapixcel(width, height, img_mp_10)
            elif output_file_format == 'png' and width * height > img_mp_25 * 1000000 and client == 'windows':
                flag = True
                width, height = resize_megapixcel(width, height, img_mp_25)
            elif output_file_format == 'png' and width * height > img_mp_25 * 1000000 and client == 'mac':
                flag = True
                width, height = resize_megapixcel(width, height, img_mp_25)
            elif output_file_format == 'jpg' and width * height > img_mp_25 * 1000000:
                flag = True
                width, height = resize_megapixcel(width, height, img_mp_25)
        else:
            megapixcel_point_25 = 0.25
            if get_megapixcel(width, height) > 0.25:
                flag = True
                width, height = resize_megapixcel(width, height, megapixcel_point_25)

        exif = source_image_pil._getexif()
        if exif is not None:
            infos = dict(exif.items())
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    if orientation in infos:
                        if infos[orientation] == 3:
                            source_image_pil = source_image_pil.rotate(180, expand=1)
                        elif infos[orientation] == 6:
                            source_image_pil = source_image_pil.rotate(270, expand=1)
                            flag = True
                            width, height = height, width
                        elif infos[orientation] == 8:
                            source_image_pil = source_image_pil.rotate(90, expand=1)
                            flag = True
                            width, height = height, width
                        break
        if flag:
            # print(width,height)
            source_image_pil = source_image_pil.resize((width, height), Image.ANTIALIAS)

        if source_image_pil.mode == 'P':
            source_image_pil = source_image_pil.convert('RGB')
        if source_image_pil.mode == 'RGBA':
            source_image_pil = source_image_pil.convert('RGB')
        return source_image_pil
    except Exception as e:
        print('==== Error Start : func name -> upload_source_image_file =====')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('==== Error End : func name -> upload_source_image_file =====')
        return None, None, None
    
    
    
def resize_megapixcel(width, height, megapixcel):
    width_ratio = width / 1000
    height_ratio = height / 1000
    ratio_mul = width_ratio * height_ratio
    width = width * math.sqrt(megapixcel * 1000000 / ratio_mul)
    width = (int)(width / 1000)
    height = height * math.sqrt(megapixcel * 1000000 / ratio_mul)
    height = (int)(height / 1000)
    return width, height


def get_megapixcel(width, height):
    return (width * height) / 1000000


def get_current_time_in_millis():
    millis = int(round(time.time() * 1000))
    return millis


def getExif_img(img):
    exif = img._getexif()
    if exif is not None:
        infos = dict(exif.items())

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                if orientation in infos:
                    if infos[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif infos[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif infos[orientation] == 8:
                        img = img.rotate(90, expand=True)
                    break
    return img


def is_url_image(url):
    try:
        if 'drive.google.com/file/d/' in url:
            url = 'https://drive.google.com/uc?id=' + url.rsplit('/', 2)[1]
        download_image_path = '/opt/gdrive/'
        if 'drive.google' in url:
            # if config.environment is 'prod':
            #     proxy = 'http://8nyg9:135e5h23@131.153.133.18:6007'
            # else:
            proxy = 'http://8nyg9:135e5h23@131.153.133.18:6007'
            gdrive_img = gdown.download(url, download_image_path, quiet=True, proxy=proxy)
            img = Image.open(gdrive_img)
            source_image_file_size = os.stat(gdrive_img).st_size
            source_image_file_size_mb = source_image_file_size / 1000000
            if os.path.exists(gdrive_img):
                os.remove(gdrive_img)

        else:
            if 'dropbox' in url:
                url = url.replace("dl=0", "dl=1")
            agent = {
                "User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}

            img_data = requests.get(url, headers=agent)
            if 'Content-Length' in img_data.headers:
                length = int(img_data.headers['Content-Length'])
                source_image_file_size_mb = length / 1000000
                img = Image.open(BytesIO(img_data.content))
                img = getExif_img(img)
            else:
                img_data = requests.get(url, headers=agent).content
                img = Image.open(BytesIO(img_data))
                # contents = img.getvalue()
                length = len(img_data)
                source_image_file_size_mb = length / 1000000
                # print(source_image_file_size_mb)
                img = getExif_img(img)
        return img, source_image_file_size_mb
    except Exception as e:
        return None, None


def get_mask(output_img_path, source_image_name):
    uuid = get_uuid()
    mask_img_name = source_image_name.rsplit('.', 1)[0] + '.png'
    img = Image.open(output_img_path)
    alpha = img.getchannel(3)
    mask_path = os.path.join(IMAGE_DOWNLOAD_FOLDER_PATH, uuid, mask_img_name)
    alpha.save(mask_path)
    return mask_path


def get_current_time_in_millis():
    millis = int(round(time.time() * 1000))
    return millis

def get_uuid():
    uid = str(uuid.uuid1())
    if not os.path.exists(IMAGE_DOWNLOAD_FOLDER_PATH + uid):
        os.makedirs(IMAGE_DOWNLOAD_FOLDER_PATH + uid)
    return uid
