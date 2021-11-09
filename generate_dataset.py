import os
from layeris.layer_image import LayerImage
import numpy as np
import csv
import cv2

ORIG_IMG_DIR = "/Users/michaelchang/Documents/cs230/val/val_256/"
NEW_DIR = "/Users/michaelchang/Documents/cs230/val/val_256_edited/"


def main():
    i = 0
    header = ['filename', 'brightness', 'contrast', 'saturation', 'lightness']
    f = open('val/corrections.csv', 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(header)

    for filename in os.listdir(ORIG_IMG_DIR):
        image = LayerImage.from_file(os.path.join(ORIG_IMG_DIR, filename))
        if image.get_image_as_array().shape != (256, 256, 3):
            continue
        #image = LayerImage.from_file(os.path.join(ORIG_IMG_DIR, str()))
        
        #Randomly sample values for brightness, contrast, saturation, and lightness
        brightness = np.random.normal(0, 0.2)
        contrast = max(0.1, np.random.normal(1, 0.4))
        saturation = np.random.normal(0, 0.4)
        lightness = np.random.normal(0, 0.1)

        #Apply the image edits
        image.brightness(brightness)
        image.contrast(contrast)
        image.saturation(saturation)
        image.lightness(lightness)

        #add guassian noise
        image = noisy("gauss", image)

        row_data = [filename, brightness, contrast, saturation, lightness]
        writer.writerow(row_data)

        image.save(os.path.join(NEW_DIR, filename))
            
    f.close()

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

if __name__ == '__main__':
	main()