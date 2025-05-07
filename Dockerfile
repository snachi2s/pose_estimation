FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0

# workdir
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-tk \
    libopenblas-dev \
    libfreetype6-dev \
    pkg-config \
    gfortran \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    ffmpeg \
    imagemagick \
    x11-apps \
    feh \
    && rm -rf /var/lib/apt/lists/*

#symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt 

#copy all the script
COPY *.py ./

RUN mkdir -p /app/data /app/output
COPY data/* /app/data/

#welcome script with instructions
RUN echo '#!/bin/bash' > /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    echo 'echo "===== 3D Box Detection Application ====="' >> /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    echo 'echo "To run the application, execute:"' >> /app/welcome.sh && \
    echo 'echo "    python main.py"' >> /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    echo 'echo "The transformation matrix will be saved to:"' >> /app/welcome.sh && \
    echo 'echo "    /app/output/camera_to_box_transform.npy"' >> /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    echo 'echo "To view the visualization directly, run:"' >> /app/welcome.sh && \
    echo 'echo "    python main.py  # This will display the plot window"' >> /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    echo 'echo "To view saved images, run:"' >> /app/welcome.sh && \
    echo 'echo "    feh /app/output/box_detection_result.png"' >> /app/welcome.sh && \
    echo 'echo ""' >> /app/welcome.sh && \
    chmod +x /app/welcome.sh

ENTRYPOINT ["/bin/bash", "-c", "/app/welcome.sh && /bin/bash"]

VOLUME ["/app/output"]