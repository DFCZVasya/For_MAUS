from yolodetect import Conductor, YoloThread
from threading import Thread

obj = Conductor()
Thread(target=YoloThread, args=(obj,)).start()
while True:
    print(obj.get_dict())
