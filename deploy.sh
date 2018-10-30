scp client.py pi:/home/pi/mingyu/can-selfdriving/
scp config.py pi:/home/pi/mingyu/can-selfdriving/
scp car.py pi:/home/pi/mingyu/can-selfdriving/
scp -r ./util pi:/home/pi/mingyu/can-selfdriving/
scp -r ./control pi:/home/pi/mingyu/can-selfdriving/
scp -r ./fcn pi:/home/pi/mingyu/can-selfdriving/
rsync -av -e ssh --exclude='*.png' ./tests pi:/home/pi/mingyu/can-selfdriving/
