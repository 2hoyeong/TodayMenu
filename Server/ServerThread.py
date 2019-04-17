import queue
import threading
import Server

class ServerThread():
    def __init__(self):
        self.clientQ = queue.Queue()
        #self.clientDict = dict()

    def append(self, client):
        self.clientQ.put(client)



