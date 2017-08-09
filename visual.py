# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import caffe

deploy_file = "/Users/lhw/caffe-project/CNN_age_gender/deploy_gender.prototxt"
model_file = "/Users/lhw/caffe-project/CNN_age_gender/gender_net.caffemodel"
image_file = "/Users/lhw/caffe-project/CNN_age_gender/example_image.jpg"
feature_map_path = "/Users/lhw/caffe-project/CNN_age_gender/draw_data/"


def show_data(data, name, padsize=1, padval=0):
    # 归一化
    data -= data.min()
    data /= data.max()
    # 根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize))

    data = np.pad(data, padding, mode='constant', constant_values=padval)

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3))

    data = data.reshape((n * data.shape[1], n * data.shape[3]))

    image_path = os.path.join(feature_map_path, name)
    # plt.set_cmap('gray')
    plt.imsave(image_path, data)
    #plt.axis('off')

    """
    print name
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()
    """

net = caffe.Net(deploy_file, model_file, caffe.TEST)
# 输出网络每一层的参数形状
print [(k, v[0].data.shape) for k, v in net.params.items()]
# 图像预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# 导入图片
image = caffe.io.load_image(image_file)
# 执行上面设置的图像预处理操作，并将图片载入blob中
net.blobs['data'].data[...] = transformer.preprocess('data', image)
# 前向迭代，即分类，保存输出
out = net.forward()
# 输出结果为各个可能分类的概率分布
print "prob: ", out['prob']
# 最可能分类
predict = out['prob'].argmax()
print "Result: ", str(predict)

feature = net.blobs['conv1'].data
show_data(feature.reshape(96, 56, 56), 'conv1.jpg')

feature = net.blobs['pool1'].data
show_data(feature.reshape(96, 28, 28), 'pool1.jpg')

feature = net.blobs['conv2'].data
show_data(feature.reshape(256, 28, 28), 'conv2.jpg')

feature = net.blobs['pool2'].data
show_data(feature.reshape(256, 14, 14), 'pool2.jpg')

feature = net.blobs['conv3'].data
show_data(feature.reshape(384, 14, 14), 'conv3.jpg')

feature = net.blobs['pool5'].data
show_data(feature.reshape(384, 7, 7), 'pool5.jpg')

