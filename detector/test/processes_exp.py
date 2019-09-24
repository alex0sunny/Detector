from multiprocessing import Process
import time


def f(name=None, tout=1):
    if not name:
        name = 'Noname'
    while True:
        print('hello', name)
        time.sleep(tout)


if __name__ == '__main__':
    p = Process(target=f)
    p.start()
    p2 = Process(target=f, args=('Bob', .7))
    p2.start()
    #p.join()
    time.sleep(5)
    p.terminate()
    p2.terminate()

