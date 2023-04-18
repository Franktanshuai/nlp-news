# 基础镜像
FROM python:3.7-slim-buster

# 设置工作目录
WORKDIR /app

# 将当前目录下的文件拷贝到工作目录
COPY . /app

# 安装应用程序所需的Python库
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 5000

# 启动Flask应用程序
CMD ["python", "app.py"]

