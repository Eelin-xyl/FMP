from multiprocessing import Process, Queue, Pipe
import queue
import time


def worker(interval, queue, recv_port):
    n = 5
    while n > 0:
        # t, a = queue.get()
        r = recv_port.recv()
        # print(t)
        # print(a)
        print("The time is {0}".format(time.ctime()))
        print(r)
        time.sleep(interval)
        n -= 1


if __name__ == "__main__":

    queue = Queue()
    queue.put(['test1', '1'])
    send_port, recv_port = Pipe()
    send_port.send('test2')

    # send_port.send('test4')
    # send_port.send('test5')
    p = Process(target=worker, args=(2, queue, recv_port))
    send_port.send('test3')
    # p.daemon = True
    p.start()
    print("p.pid:", p.pid)
    print("p.name:", p.name)
    print("p.is_alive:", p.is_alive())
