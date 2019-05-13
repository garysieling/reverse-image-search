FROM apache/zeppelin:0.8.1
RUN apt-get -y install python3-pip
RUN pip3 install seaborn pandas matplotlib bokeh xarray mxnet
RUN apt-get -y install ffmpeg
RUN pip3 install --upgrade youtube_dl
RUN pip3 install opencv-python
COPY interpreter.json /zeppelin/conf/interpreter.json
