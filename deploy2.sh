scp client.py pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp config.py pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp car-avoid.py pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp car.py pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp -r ./util pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp -r ./control pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
scp -r ./fcn pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
rsync -av -e ssh --exclude='*.png' ./tests pi@192.168.1.5:/home/pi/mingyu/can-selfdriving/
# scp pi@192.168.1.5:/home/pi/Desktop/car/autonomous_vehicles_canlab_copy/car1/processImage.py ./control
