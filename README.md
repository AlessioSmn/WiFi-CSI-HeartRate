# WiFi-CSI-HeartRate
Group project for Industrial Application course, MSc in Computer Engineering, Università di Pisa, A.Y. 2025/2026
## Description
## Repository structure
## Hardware
Raspberry Pi 3 Model B+

ESP-WROOM-32 (x3)

MAX30102 sensor

STM32F4-DISCOVERY

## Installation
### Linux (PC x86)
Create a new virtual environment for training the LSTM model and converting it with TensorFlow Lite:
```bash
conda create -n tflite python=3.11
conda activate tflite
```
Install the following packages:
```bash
pip install pandas==2.3.3
pip install numpy==1.23.5
pip install tensorflow==2.12.0
pip install tflite-runtime==2.14.0
```

### Raspberry
Create a new virtual environment for training the LSTM model and converting it with TensorFlow Lite:
```bash
conda create -n tflite python=3.11
conda activate tflite
```
Install the following packages:
```bash
pip install pandas==3.0.0
pip install numpy==1.26.4
pip install pyserial==3.5
pip install scipy==1.17.0
pip install tflite-runtime==2.14.0
```

## Deployment
### Raspberry
```bash
python hr_detection_lite_parallel.py -p /dev/ttyUSB0 -ps /dev/ttyACM0
```
