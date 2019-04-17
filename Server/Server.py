import threading
import time
import Header
import PacketHandler

class Server(threading.Thread):
    def __init__(self, conn, addr):
        threading.Thread.__init__(self)
        self.conn = conn
        self.addr = addr

    def run(self):
        while True:
            data = self.conn.recv(1024).decode('euc-kr')
            
            if not data:
                print("CONNECTION CLOSE")
                break;

            header, data = data.split("|", 1)
            PacketHandler.PacketHandler().switch(self, header, data)

    def send(self, header, data):
        message = bytes((str(header) + "|" + data), 'euc-kr')
        self.conn.send(message)

