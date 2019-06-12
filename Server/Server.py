import threading
import time
import Header
from PacketHandler import PacketHandler

class Server(threading.Thread):
    def __init__(self, conn, addr):
        threading.Thread.__init__(self)
        self.conn = conn
        self.addr = addr
        self.id = None
        self.key = None

    def run(self):
        while True:
            data = self.conn.recv(1024).decode('utf-8')
            
            if not data:
                print("CONNECTION CLOSE")
                break;

            data = data[:-1]

            try:
                header, data = data.split("|", 1)
                PacketHandler().switch(self, header, data)
            except Exception as err:
                print(data)
                print(err)

    def send(self, header, data):
        data += "\n";
        message = bytes((str(header.value) + "|" + data), 'utf-8')
        print(message[:-1].decode('utf-8'))
        self.conn.send(message)

    def setId(self, id):
        self.id = id
