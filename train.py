import Deal_Image
import tensorflow as tf
from Deal_Image import *
from Find_bound import *
import cv
x_start = 0
y_start = 0
half_height = 350
half_width =  400

number_width = 12
number_heigh = 28


def train(if_train):
    sess = tf.InteractiveSession()
    img =  get_image(sess)
    up_img = crop_digits(sess,img,x_start,y_start,half_height,half_width)
    get_bound_rects(up_img)
    #plot_sampleWithLables(img, 1000, sess, tartget_height, target_width)







    #startx = 0
    #starty = 0
    #for i in range(8):
     #   number = crop_digits(sess,img,startx,starty,number_heigh,number_width)
      #  plot_sampleWithLables(number,i,sess,number_heigh,number_width)
       # startx += number_width



def main(argv=None):
    train(if_train=True)




if __name__ == '__main__':
  tf.app.run()