import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
INPUT_IMAGE_DIM = 9216
IMAGE_DIR = 'data/xs.JPG'



def get_image(sess):
    filename_queue = tf.train.string_input_producer([IMAGE_DIR])  # list of files to read
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value,channels=1)
    #img = tf.image.crop_to_bounding_box(img,)
    #resized_image = tf.image.resize_images(image, 654, 493)
    # image_batch = tf.train.batch([my_img], batch_size=1)
    #resized_image = tf.reshape(img, [1, INPUT_IMAGE_DIM])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image = image.eval()  # here is your image Tensor :)
    coord.request_stop()
    coord.join(threads)
    return image


def crop_digits(sess,img,x_start,y_start,target_height,target_width):
    image =  tf.image.crop_to_bounding_box(img,y_start,x_start,target_height,target_width)
    image = image.eval()  # here is your image Tensor :)
    return image

def plot_sampleWithLables(img,i,sess,height,width):

    img = img.reshape(height, width)


    plt.imshow(img, cmap='gray')

    plt.savefig("data/img" +str(i)  +".png")
    return





