import tensorflow as tf
import numpy as np
import cv2 as cv


# 使用OpenCV显示图像
def show_image(image, title='input'):
    print("result : \n", image)
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


sess = tf.Session()
x_input = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]],np.int32)
print("x_input.shape : ", x_input.shape)
x_input = np.expand_dims(x_input, 0)
x_input = np.expand_dims(x_input, 3)
print("expand x_input.shape : ", x_input.shape)

resized = tf.image.resize_nearest_neighbor(x_input, size=[8, 8])
resized = tf.squeeze(resized)
result = sess.run(resized)

result = np.uint8(result)
show_image(result, 'input data')

src = cv.imread("D:/vcprojects/images/meinv.png")
cv.imshow("input", src)
h, w, depth = src.shape
src = np.expand_dims(src, 0)
print(src.shape)

bi_image = tf.image.resize_bilinear(src, size=[h*2, w*2])
bi_image = tf.squeeze(bi_image)
bi_result = sess.run(bi_image)
bi_result = np.uint8(bi_result)
show_image(bi_result,"bilinear-zoom")

src = cv.imread("D:/vcprojects/images/meinv.png")
src = np.expand_dims(src, 0)
brightness = tf.image.adjust_brightness(src, delta=.5)
brightness = tf.squeeze(brightness)
result = sess.run(brightness)
result = np.uint8(result)
show_image(result, "brightness demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
src = np.expand_dims(src, 0)
contrast = tf.image.adjust_contrast(src, contrast_factor=2.2)
contrast = tf.squeeze(contrast)
result = sess.run(contrast)
result = np.uint8(result)
show_image(result, "contrast demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
src = np.expand_dims(src, 0)
contrast = tf.image.adjust_gamma(src, gain=1.0, gamma=4.2)
contrast = tf.squeeze(contrast)
result = sess.run(contrast)
result = np.uint8(result)
show_image(result, "gamma demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
contrast = tf.image.adjust_saturation(src, saturation_factor=2.2)
result = sess.run(contrast)
result = np.uint8(result)
show_image(result, "saturation demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
contrast = tf.image.per_image_standardization(src)
result = sess.run(contrast)
result = np.uint8(result)
show_image(result, "standardization demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
#src = np.expand_dims(src, 0)
contrast = tf.image.resize_image_with_crop_or_pad(src, np.int32(h*1.5), np.int32(w*1.5))
#contrast = tf.squeeze(contrast)
result = sess.run(contrast)
result = np.uint8(result)
show_image(result, "resize and crop or pad - demo")

src = cv.imread("D:/vcprojects/images/meinv.png")
gray = tf.image.rgb_to_grayscale(src)
result = sess.run(gray)
result = np.uint8(result)
show_image(result, "gray - demo")


jpg = tf.read_file("D:/vcprojects/images/yuan_test.png")
img = tf.image.decode_jpeg(jpg, channels=3)
cropped = tf.random_crop(img, [500, 400, 3])
cropped_run = sess.run(cropped)
result = cv.cvtColor(cropped_run, cv.COLOR_RGB2BGR)
result = np.uint8(result)
show_image(result, "random-crop-demo")




