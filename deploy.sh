scp client.py canpi:/home/pi/mingyu/can-selfdriving/
scp config.py canpi:/home/pi/mingyu/can-selfdriving/
scp car-avoid.py canpi:/home/pi/mingyu/can-selfdriving/
scp car.py canpi:/home/pi/mingyu/can-selfdriving/
scp -r ./util canpi:/home/pi/mingyu/can-selfdriving/
scp -r ./control canpi:/home/pi/mingyu/can-selfdriving/
scp -r ./fcn canpi:/home/pi/mingyu/can-selfdriving/
rsync -av -e ssh --exclude='*.png' ./tests canpi:/home/pi/mingyu/can-selfdriving/
# scp pi@192.168.1.5:/home/pi/Desktop/car/autonomous_vehicles_canlab_copy/car1/processImage.py ./control
