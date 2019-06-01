# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import mxnet as mx

ctx = mx.cpu()
from mxnet.gluon.model_zoo.vision import mobilenet1_0

pretrained_net = mobilenet1_0(pretrained=True)
print(pretrained_net)

checkpoint = "/mnt/85b35634-4007-4f18-a6b3-5f6d47385a82/projects/reverse-image-search/evaluation/confusion/data/checkpoints/9-0.params"
net = mobilenet1_0(classes=3)

from mxnet import init
net.features = pretrained_net.features
net.output.initialize(init.Xavier())
net.collect_params().reset_ctx(ctx)

from mxnet.image import color_normalize
from mxnet import image

test_augs = [
    image.ResizeAug(224),
    image.CenterCropAug((224, 224))
]

labels = ["breaker_box", "sump_pump", "washer"]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = mx.nd.transpose(data, (2, 0, 1))
    return data, mx.nd.array([label]).asscalar().astype('float32')


def predict_class(net, img):
    # with open(fname, 'rb') as f:
    # img = image.imdecode(f.read())
    data, _ = transform(img, -1, test_augs)
    # plt.imshow(data.transpose((1,2,0)).asnumpy()/255)
    data = data.expand_dims(axis=0)
    data = color_normalize(data / 255,
                           mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)),
                           std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)))
    out = net(data.as_in_context(mx.cpu()))
    # plt.imshow(img.asnumpy())
    pred, label = get_label_and_prod(out)
    return label
    # print('Pred: %s'% label)


def get_label_and_prod(out):
    pred = int(mx.nd.argmax(out, axis=1).asscalar())
    return pred, labels[pred]


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

predictions = {}

for label in labels:
    predictions[label] = 0

# loop over frames from the video file stream
while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()
    if frame is None:
        continue

    frame = imutils.resize(frame, width=1800)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    prediction = predict_class(net, mx.nd.array(frame))
    print(prediction)

    predictions[prediction] = predictions[prediction] + 1

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print(predictions)

max = 0
total = 0
for key, value in predictions.items():
    if value > max:
        max = value

    total = total + value

percent = 100 * max / total
print(percent)

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
