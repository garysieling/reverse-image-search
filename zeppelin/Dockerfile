FROM apache/zeppelin:0.8.1
RUN apt-get -y update && apt-get -y upgrade

# needed so pip doesn't make matplotlib think python is too old
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.7
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.7

RUN pip3.7 install seaborn pandas matplotlib bokeh xarray

# ERROR: mxnet 1.4.1 has requirement numpy<1.15.0,>=1.8.2, but you'll have numpy 1.16.3 which is incompatible
# it looks like this next one will forcibly downgrade numpy
RUN pip3.7 install mxnet
RUN apt-get -y install ffmpeg
RUN pip3.7 install --upgrade youtube_dl
RUN pip3.7 install opencv-python
RUN apt-get -y install jq
COPY interpreter.json /zeppelin/conf/interpreter.json
COPY zeppelin-site.xml /zeppelin/conf/zeppelin-site.xml
