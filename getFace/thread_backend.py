import threading, time


class Seeker(threading.Thread):
    def __init__(self, cond, name):
        threading.Thread.__init__(self)
        self.cond = cond
        self.name = name

    def run(self):
        time.sleep(1)  # 1.确保seeker晚于hider开始执行

        self.cond.acquire()  # 4. hider的锁释放了所以这里获得了锁
        print('我把眼睛蒙上了')
        self.cond.notify()  # 5.蒙上眼后通知hider，hider线程此时被唤醒并试图获取锁，但是锁还在seeker身上，所以hider被阻塞，seeker继续往下
        self.cond.wait()  # 6. seeker锁被释放并且挂起，hider就获取锁开始继续往下运行了

        print('我找到你了')
        self.cond.notify()  # 9.找到了之后通知hider，hider意图获取锁但不行所以被阻塞，seeker往下
        self.cond.release()  # 10.释放锁

        print('我赢了')


class Hider(threading.Thread):
    def __init__(self, cond, name):
        threading.Thread.__init__(self)
        self.cond = cond
        self.name = name

    def run(self):
        self.cond.acquire()  # 2.hider获取锁
        self.cond.wait()  # 3.hider被挂起然后释放锁

        print('我已经藏好了')
        self.cond.notify()  # 7.藏好后通知seeker，seeker意图获取锁，但是锁在hider身上所以seeker被阻塞
        self.cond.wait()  # 8.hider被挂起，释放锁，seeker获取锁，seeker继续往下运行

        self.cond.release() # 11. 在此句之前一点，seeker释放了锁（#10），hider得到锁，随即这句hider释放锁
        print ('被你找到了')


cond = threading.Condition()
seeker = Seeker(cond, 'seeker')
hider = Hider(cond, 'hider')
seeker.start()
hider.start()