FROM apache/zeppelin:0.8.1
RUN apt-get -y install python3-pip
RUN pip3 install seaborn pandas matplotlib bokeh xarray
COPY interpreter.json /zeppelin/conf/interpreter.json
