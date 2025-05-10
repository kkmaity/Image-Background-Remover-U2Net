import time

import cv2
import numpy as np
import torch
from PIL import Image
from cv2 import dnn_superres
from torchvision import transforms


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])




def getTrimpap(inputMask, size, minThresh, conf_threshold):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [3]: an image with probabilities of each pixel being the foreground, size of dilation kernel,
    foreground confidence threshold
    Output    : a trimap
    """
    # mask = inputMask.copy()
    mask = np.zeros_like(inputMask)
    mask[inputMask > minThresh] = 255
    mask[inputMask > conf_threshold] = 0

    pixels = 2 * size + 1
    kernel = np.ones((pixels, pixels), np.uint8)

    dilation = cv2.dilate(mask, kernel, iterations=1)
    # dilation = cv2.erode(dilation, kernel, iterations=1)

    # dilation = cv2.erode(dilation, kernel, iterations=1)
    # dilation = cv2.dilate(dilation, kernel, iterations=2)

    remake = np.zeros_like(mask)
    remake[dilation == 255] = 127  # Set every pixel within dilated region as probably foreground.
    remake[inputMask > conf_threshold] = 255  # Set every pixel with large enough probability as definitely foreground.
    # remake[inputMask == 0] = 0

    return remake


def get_cassification(img):
    img_t = data_transforms(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = vgg_based(batch_t)
    # ['animals', 'bike', 'car', 'cycles', 'graphics', 'human', 'product', 'realestate']

    _, index = torch.max(out, 1)
    val = int(index[0].numpy())
    if val == 0:
        print('animal')
        return 'animal'
    if val == 1:
        print('bike')
        return 'bike'
    if val == 2:
        print('car')
        return 'car'
    if val == 3:
        print('cycle')
        return 'cycle'
    if val == 4:
        print('graphics')
        return 'graphics'
    if val == 5:
        print('human')
        return 'human'
    if val == 6:
        print('product')
        return 'product'
    if val == 7:
        print('realestate')
        return 'realestate'


def getSemi(mask, thresh):
    semiPos = (mask > 0)
    semiPos = np.multiply(semiPos, 1)

    semiNeg = (mask < thresh)
    semiNeg = np.multiply(semiNeg, 1)

    semi = cv2.bitwise_and(semiPos, semiNeg)

    notSemi = cv2.bitwise_not(semi) + 2

    return semi, notSemi


def pilresize(im, size):
    pil = Image.fromarray(im).resize(size, resample=Image.BILINEAR)
    return np.asarray(pil)


def correctColorPy(originalImg, originalMask):
   
    details_filter_size = 15
    first_iteration_filter_size = 3
    other_iterations_filter_size = 7

    originalMask = originalMask.astype(np.uint8)
    originalMask[originalMask < 5] = 0


    start = time.time()

    width = originalMask.shape[1]
    height = originalMask.shape[0]

    originalMask = originalMask.astype(np.float32) * 1.008
    originalMask = np.clip(originalMask, a_min=0, a_max=255)
    originalMask = originalMask.astype(np.uint8)

    opaque = 255
    originalSemi, originalNotSemi = getSemi(originalMask, opaque)



    if (originalImg.shape[0] == originalMask.shape[0] and originalImg.shape[1] == originalMask.shape[1]):
        # if the image width is bigger than 600 will are down-scalling the image
        if width > 600:
            new_width = 600
            new_height = int(height * (600 / width))

            dim = (new_width, new_height)
            img = pilresize(originalImg, dim)
            mask = pilresize(originalMask, dim)
            
            img = originalImg.copy()
            mask = originalMask.copy()
        else:
            img = originalImg.copy()
            mask = originalMask.copy()

        floatMask = mask.copy()

        # get the parts of the mask were is between 0 and 255 in semi and the rest of the mask in notSemi
        semi, notSemi = getSemi(mask, opaque)
        semi = semi.astype(np.float32)
        notSemi = notSemi.astype(np.float32)

        # I have found that processing each channel of the image separately to be faster even though the same processing is
        # applied to each of them
        r, g, b = cv2.split(img)
        outputR = r.astype(np.float32)
        outputG = g.astype(np.float32)
        outputB = b.astype(np.float32)

        # for the next part we are binarizing the mask to 0 and 1
        floatMask[floatMask < opaque] = 0
        floatMask[floatMask > 0] = 1

        # here we are separating the part of the image where the mask is 255
        maskedR = outputR * floatMask.astype(np.float32)
        maskedG = outputG * floatMask.astype(np.float32)
        maskedB = outputB * floatMask.astype(np.float32)

        # to fill parts of the image corresponding to the semi transparent parts of the mask we are first
        # applying this filter
        kernel = np.ones((first_iteration_filter_size, first_iteration_filter_size), np.float32)

        # here we are counting the number of semi transparent pixels in the mask
        # so later on when we fill these pixels we will be updating this number untill it become 0
        semi_white_pix = np.sum(semi == 1)
        initial_white_pix = semi_white_pix

        uintMask = []
        not_mask = []
        for i in range(50):
            # print(i)
            # fill the empty parts using a mean filter by excluding the 0 pixels
            fmaskedR = cv2.filter2D(maskedR, -1, kernel)
            fmaskedG = cv2.filter2D(maskedG, -1, kernel)
            fmaskedB = cv2.filter2D(maskedB, -1, kernel)
            floatMask2 = cv2.filter2D(floatMask.astype(np.float32), -1, kernel)

            fmaskedR /= (floatMask2 + 0.00001)
            fmaskedG /= (floatMask2 + 0.00001)
            fmaskedB /= (floatMask2 + 0.00001)
            # floatMask2 /= (floatMask2 + 0.00001)

            # update the images for each iteration
            floatMask2[floatMask2 > 0] = 1
            floatMask2 *= semi.astype(np.float32)
            fmaskedR *= floatMask2.astype(np.float32)
            fmaskedG *= floatMask2.astype(np.float32)
            fmaskedB *= floatMask2.astype(np.float32)

            maskedR *= floatMask2.astype(np.float32)
            maskedG *= floatMask2.astype(np.float32)
            maskedB *= floatMask2.astype(np.float32)

            if i > 0:
                uintMask = uintMask.astype(np.float32)
                not_mask = not_mask.astype(np.float32)
                not_mask *= semi.astype(np.float32)

                maskedR = (maskedR * uintMask) + (fmaskedR * not_mask)
                maskedG = (maskedG * uintMask) + (fmaskedG * not_mask)
                maskedB = (maskedB * uintMask) + (fmaskedB * not_mask)
            else:
                maskedR = fmaskedR.copy()
                maskedG = fmaskedG.copy()
                maskedB = fmaskedB.copy()

            maskedR *= floatMask2.astype(np.float32)
            maskedG *= floatMask2.astype(np.float32)
            maskedB *= floatMask2.astype(np.float32)

            uintMask = floatMask2.copy()
            uintMask = uintMask.astype(np.uint8)
            not_mask = cv2.bitwise_not(uintMask) + 2

            floatMask = floatMask2
            kernel = np.ones((other_iterations_filter_size, other_iterations_filter_size), np.float32)

            # count the number of empty pixel and check if there still any
            current_white_pix = np.sum(floatMask == 1)
            changed = initial_white_pix - current_white_pix

            if (changed <= 0):
                break
            else:
                semi_white_pix -= changed

        floatMask = floatMask.astype(np.uint8)
        notFloatMask = cv2.bitwise_not(floatMask) + 2
        maskedR = (maskedR * floatMask.astype(np.float32)) + (outputR * notFloatMask.astype(np.float32))
        maskedG = (maskedG * floatMask.astype(np.float32)) + (outputG * notFloatMask.astype(np.float32))
        maskedB = (maskedB * floatMask.astype(np.float32)) + (outputB * notFloatMask.astype(np.float32))

        # merge the semi transparent part with the rest of the image
        outputR = (maskedR * semi) + (outputR * notSemi)
        outputR = outputR.astype(np.uint8)

        outputG = (maskedG * semi) + (outputG * notSemi)
        outputG = outputG.astype(np.uint8)

        outputB = (maskedB * semi) + (outputB * notSemi)
        outputB = outputB.astype(np.uint8)

        # re-construct the rgb image
        outputImage = cv2.merge((outputR, outputG, outputB))

        # if the input image width is bigger than 600 go back to the original resolution
        if width > 600:
            outputImage = sr4.upsample(outputImage)
            dim = (width, height)
            # outputImage = cv2.resize(outputImage, dim, interpolation=cv2.INTER_AREA)
            outputImage = pilresize(outputImage, dim)

        r, g, b = cv2.split(originalImg)
        our, oug, oub = cv2.split(outputImage)

        # get the local details of the images
        kernel = np.ones((details_filter_size, details_filter_size), np.float32) / (
                details_filter_size * details_filter_size)
        """
        blurredR = cv2.filter2D(r.astype(np.float32), -1, kernel)
        blurredG = cv2.filter2D(g.astype(np.float32), -1, kernel)
        blurredB = cv2.filter2D(b.astype(np.float32), -1, kernel)


        detailsR = r.astype(np.float32) - blurredR.astype(np.float32)
        detailsG = g.astype(np.float32) - blurredG.astype(np.float32)
        detailsB = b.astype(np.float32) - blurredB.astype(np.float32)
        """
        our = our.astype(np.float32)
        oug = oug.astype(np.float32)
        oub = oub.astype(np.float32)

        gray = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
        blurredGray = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        detailsGray = gray.astype(np.float32) - blurredGray.astype(np.float32)

        detailsR = detailsGray * (our / 255)
        detailsG = detailsGray * (oug / 255)
        detailsB = detailsGray * (oub / 255)

        our += detailsR
        oug += detailsG
        oub += detailsB

        kernel = np.ones((5, 5), np.float32) / 25
        originalSemi = cv2.filter2D(originalSemi.astype(np.float32), -1, kernel)
        originalNotSemi = cv2.filter2D(originalNotSemi.astype(np.float32), -1, kernel)

        our = (our * originalSemi) + (r.astype(np.float32) * originalNotSemi)
        oug = (oug * originalSemi) + (g.astype(np.float32) * originalNotSemi)
        oub = (oub * originalSemi) + (b.astype(np.float32) * originalNotSemi)

        our[our < 0] = 0
        oug[oug < 0] = 0
        oub[oub < 0] = 0

        our[our > 255] = 255
        oug[oug > 255] = 255
        oub[oub > 255] = 255

        # outputImage = cv2.merge((oub.astype(np.uint8), oug.astype(np.uint8), our.astype(np.uint8)))
        orig = cv2.merge((our.astype(np.uint8), oug.astype(np.uint8), oub.astype(np.uint8)))

        end = time.time()
        print("TIme = " + str(end - start))
        # orig = cv2.merge((our.astype(np.uint8), oug.astype(np.uint8), oub.astype(np.uint8)))
        orig = Image.fromarray(orig)
        # orig.show()
        mask = Image.fromarray(originalMask).convert('L')
        orig.putalpha(mask)
        # orig.show()
        return orig, mask


sr4 = dnn_superres.DnnSuperResImpl_create()
sr4.readModel("ESPCN_x4.pb")
sr4.setModel("espcn", 4)

def apply_mask_trimap(orig, mask):
    # Image.fromarray(mask).show()
    original_img_width, original_img_height = orig.size
    original_img_dim = original_img_width if original_img_width > original_img_height else original_img_height
    print("Without matting")
    mask = sr4.upsample(mask)
    mask_height = mask.shape[0]
    mask_width = mask.shape[1]
    mask_img_dim = mask_width if mask_width > mask_height else mask_height
    if original_img_dim / mask_img_dim >= 1.5:
        mask = sr4.upsample(mask)

    mask = Image.fromarray(mask).resize(orig.size, resample=Image.BILINEAR)
    mask = np.asarray(mask)
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask[mask <48] = mask[mask <48]*1.5
    mask = mask.astype(np.uint8)
    orig = np.asarray(orig)
    orig, mask = correctColorPy(orig, mask)
    return orig, mask
