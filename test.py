# coding=utf-8

import caffe
import numpy as np

deploy = "/Users/lhw/caffe-project/race_classification/race_deploy.prototxt"
caffemodel = "/Users/lhw/caffe-project/race_classification/race_iter_500000.caffemodel"
image = "/Users/lhw/caffe-project/race_classification/test_image/aaa.bmp"

net = caffe.Net(deploy, caffemodel, caffe.TEST)

# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data',np.load(meanfile).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# 导入图片
img = caffe.io.load_image(image)
# 执行上面设置的图像预处理操作，并将图片载入blob中
net.blobs['data'].data[...] = transformer.preprocess('data', img)

out = net.forward()
race_list = ['Asian', 'Black', 'White']

prob = net.blobs['prob'].data[0].flatten()
print prob
predict = prob.argsort()[-1]
print prob.argsort()
print 'the class is', race_list[predict]

print out['prob']
predict = out['prob'].argmax()
print 'the class is', race_list[predict]

