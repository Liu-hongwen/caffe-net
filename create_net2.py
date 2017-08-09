# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P

caffe_root = "/Users/lhw/caffe-project/"
train_lmdb = caffe_root + "lmdb/train"
test_lmdb = caffe_root + "lmdb/test"
mean_file = caffe_root + "mean.binaryproto"
train_proto = caffe_root + "train.prototxt"
test_proto = caffe_root + "test.prototxt"


def create_net(lmdb, mean_file, batch_size, include_acc=False):
    # 网络规范
    net = caffe.NetSpec()

    net.data, net.label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_file=mean_file, mirror=True))

    net.conv1 = L.Convolution(net.data, num_output=96, kernel_size=11, stride=4,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0))

    net.bn1 = L.BatchNorm(net.conv1, use_global_stats=False, in_place=True,
                          param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    net.scale1 = L.Scale(net.bn1, bias_term=True, in_place=True)

    net.relu1 = L.ReLU(net.conv1, in_place=True)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    net.conv2 = L.Convolution(net.pool1, num_output=256, pad=2, kernel_size=5, group=2,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0.1))

    net.bn2 = L.BatchNorm(net.conv2, use_global_stats=False, in_place=True,
                          param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    net.scale2 = L.Scale(net.bn2, bias_term=True, in_place=True)

    net.relu2 = L.ReLU(net.conv2, in_place=True)

    net.pool2 = L.Pooling(net.conv2, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    net.conv3 = L.Convolution(net.pool2, num_output=384, pad=1, kernel_size=3,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0))

    net.bn3 = L.BatchNorm(net.conv3, use_global_stats=False, in_place=True,
                          param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    net.scale3 = L.Scale(net.bn3, bias_term=True, in_place=True)

    net.relu3 = L.ReLU(net.conv3, in_place=True)

    net.conv4 = L.Convolution(net.conv3, num_output=384, pad=1, kernel_size=3, group=2,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0.1))

    net.bn4 = L.BatchNorm(net.conv4, use_global_stats=False, in_place=True,
                          param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    net.scale4 = L.Scale(net.bn4, bias_term=True, in_place=True)

    net.relu4 = L.ReLU(net.conv4, in_place=True)

    net.conv5 = L.Convolution(net.conv4, num_output=256, pad=1, kernel_size=3, group=2,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0.1))

    net.bn5 = L.BatchNorm(net.conv5, use_global_stats=False, in_place=True,
                          param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

    net.scale5 = L.Scale(net.bn5, bias_term=True, in_place=True)

    net.relu5 = L.ReLU(net.conv5, in_place=True)

    net.pool5 = L.Pooling(net.conv5, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    net.fc6 = L.InnerProduct(net.pool5, num_output=4096,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type="gaussian", std=0.005),
                             bias_filler=dict(type="constant", value=0.1))

    net.relu6 = L.ReLU(net.fc6, in_place=True)

    net.fc7 = L.InnerProduct(net.fc6, num_output=4096,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type="gaussian", std=0.005),
                             bias_filler=dict(type="constant", value=0.1))

    net.relu7 = L.ReLU(net.fc7, in_place=True)

    net.fc8 = L.InnerProduct(net.fc7, num_output=1000,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type="gaussian", std=0.01),
                             bias_filler=dict(type="constant", value=0.1))

    net.loss = L.SoftmaxWithLoss(net.fc8, net.label)

    if include_acc:
        net.acc = L.Accuracy(net.fc8, net.label)
        return net.to_proto()

    return net.to_proto()


def write_net():
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, mean_file, batch_size=256)))

    with open(test_proto, 'w') as f:
        f.write(str(create_net(test_lmdb, mean_file, batch_size=50, include_acc=True)))


if __name__ == '__main__':
    write_net()

