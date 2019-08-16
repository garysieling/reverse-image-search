import os
from mxnet import gluon
from mxnet import nd

# TODO
# app to collect / upload tagged video
# can this model be retrained?
# lambda to retrain on upload
# do some hyperparameter tuning
# can I reassmble videos
# per video stats!
# confusion matrix complete
# nest the videos in the input set
# gpu setup (needs cuda 9.2, not 10)
# demo of basement
# reconstitution of videos

root = "/projects/reverse-image-search/evaluation/confusion"

training_map = {
    'breaker_box': [],
    'sump_pump': [],
    'washer': []
}

import copy

validation_map = copy.deepcopy(training_map)

category_to_idx = {
    'breaker_box': 0,
    'sump_pump': 1,
    'washer': 2
}


def counts(data_dir, map):
    from os.path import isfile, join

    counter = 0
    for d in os.listdir(data_dir):
        if not isfile(join(data_dir, d)):
            for root, dirs, files in os.walk(join(data_dir, d)):
                for f in files:
                    map[d].append({
                        'idx': counter,
                        'label': category_to_idx[d],
                        'filename': root + d + f,
                        'filepath': root + '/' + f
                    })
                counter += 1


counts(root + '/data/household/train/', training_map)
counts(root + '/data/household/validation/', validation_map)

print("%table type\tclass\tcount")
for k in training_map:
    print("train\t" + k + "\t" + str(len(training_map[k])))

for k in validation_map:
    print("validation\t" + k + "\t" + str(len(validation_map[k])))


import random
random.shuffle(training_map['breaker_box'])
random.shuffle(training_map['sump_pump'])
random.shuffle(training_map['washer'])

def write_lst(image_arr, base_dir, file_path):
    with open(file_path, 'w') as f:
        count = 0
        for img in image_arr:
            label = img['label']
            img_path = img['filepath']
            new_line = '\t'.join([str(count), str(label), str(img_path)])
            new_line += '\n'
            f.write(new_line)
            count += 1

# split data range
min_data_len = min(len(training_map['breaker_box']), len(training_map['sump_pump']), len(training_map['washer']))
sample = (0,8)
train = (0, int(min_data_len * 0.8))
test = (int(min_data_len * 0.80), int(min_data_len * 1))

def split_dataset(from_idx, to_idx):
    return training_map['breaker_box'][from_idx: to_idx] + training_map['sump_pump'][from_idx: to_idx] + training_map['washer'][from_idx: to_idx]

# sample set is for developing model and debugging
# because debugging with large dataset takes a long time
sample_set = split_dataset(sample[0], sample[1])
write_lst(sample_set, root + '/data/household', root + '/data/household/sample/household.lst')

train_set = split_dataset(train[0], train[1])
write_lst(train_set, root + '/data/household', root + '/data/household/train/household.lst')

validation_set = validation_map['breaker_box'] + validation_map['sump_pump'] + validation_map['washer']
write_lst(validation_set, root + '/data/household', root + '/data/household/validation/household.lst')

test_set = split_dataset(test[0], test[1])
write_lst(validation_set, root + '/data/household', root + '/data/household/test/household.lst')

from mxnet.gluon.model_zoo.vision import mobilenet1_0
pretrained_net = mobilenet1_0(pretrained=True)
print(pretrained_net)

print("classes: " + str(len(training_map)))
net = mobilenet1_0(classes=len(training_map))


from mxnet import init
net.features = pretrained_net.features
net.output.initialize(init.Xavier())


from mxnet.image import color_normalize
from mxnet import image

train_augs = [
    image.HorizontalFlipAug(0.5),
    image.BrightnessJitterAug(.3),
    image.HueJitterAug(.1)
]
test_augs = [
    image.ResizeAug(224),
    image.CenterCropAug((224, 224))
]


def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

from mxnet.gluon.data.vision import ImageRecordDataset

train_rec = root + '/data/household/train/household.rec'
validation_rec = root + '/data/household/validation/household.rec'
test_rec = root + '/data/household/test/household.rec'
sample_rec = root + '/data/household/sample/household.rec'

trainIterator = ImageRecordDataset(
    filename=train_rec,
    transform=lambda X, y: transform(X, y, train_augs)
)
validationIterator = ImageRecordDataset(
    filename=validation_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)
testIterator = ImageRecordDataset(
    filename=test_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)
sampleIterator = ImageRecordDataset(
    filename=sample_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)

import time
from mxnet.image import color_normalize
from mxnet import autograd
import mxnet as mx
from mxnet import nd


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        data = color_normalize(data / 255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)))
        output = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]


def metric_str(names, accs):
    return ', '.join(['%s=%f' % (name, acc) for name, acc in zip(names, accs)])


def train_util(net, train_iter, validation_iter, loss_fn, trainer, ctx, epochs, batch_size):
    metric = mx.metric.create(['acc'])
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_iter):
            st = time.time()
            # ensure context
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # normalize images
            data = color_normalize(data / 255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)))

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(data.shape[0])

            #  Keep a moving average of the losses
            metric.update([label], [output])
            names, accs = metric.get()
            # print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(epoch, i, batch_size/(time.time()-st), metric_str(names, accs)))
            if i % 100 == 0:
                # net.collect_params().save('/data/checkpoints/%d-%d.params'%(epoch, i))
                net.save_parameters(root + '/data/checkpoints/%d-%d.params' % (epoch, i))
        train_acc = evaluate_accuracy(train_iter, net)
        validation_acc = evaluate_accuracy(validation_iter, net)
        print("Epoch %s | training_acc %s | val_acc %s " % (epoch, train_acc, validation_acc))


def train(net, ctx,
          batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(
        trainIterator, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(
        validationIterator, batch_size)

    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})

    train_util(net, train_data, validation_data,
               loss, trainer, ctx, epochs, batch_size)


import mxnet as mx
ctx = mx.cpu()
train(net, ctx, batch_size=32, epochs=10, learning_rate=0.001)

