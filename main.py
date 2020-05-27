import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt

size = (192,192)

def loadImage():
    # load the image
    image = Image.open('test3.jpeg')
    image = image.resize(size)

    # convert image to numpy array
    data = asarray(image)
    print(type(data))
    # summarize shape
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=0)
    print(data.shape)


    return data

def plotImageWithCoordinates(coordinates):
    im = plt.imread('test3.jpeg')
    implot = plt.imshow(im)
    for coordinate in coordinates:
        x_cord = coordinate[0]  # try this change (p and q are already the coordinates)
        y_cord = coordinate[1]
        plt.scatter([x_cord], [y_cord])
    plt.show()


print('Starting model inference')

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# print('input_shape:')
# print(input_shape)

image = loadImage()


# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)

output_data = np.swapaxes(output_data,1,3)
print(output_data.shape)

for x in output_data:
    coordinates= []
    for heatmap in x:

        # TODO: implement GAUSSIAN KERNEL:
        # https://github.com/laanlabs/CarPoseDemo/blob/master/CarPoseDemo/PoseUtils.mm

        max = 0
        maxX = 0
        maxY = 0
        rows = heatmap.shape[0]
        cols = heatmap.shape[1]
        for x in range(0, rows):
            for y in range(0, cols):

                if(heatmap[x,y]>max):
                    max = heatmap[x,y]
                    maxX = x
                    maxY = y

        # TODO: implement taking imagesize from image info instead of hardcoding
        scalefactorX = 320/96
        scalefactorY = 256/96

        print('('+str(int(maxX*scalefactorX))+','+str(int(maxY*scalefactorY))+')')

        coordinates.append([maxX*scalefactorX, maxY*scalefactorY])

    plotImageWithCoordinates(coordinates)