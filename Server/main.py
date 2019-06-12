import socket
import Server
import ServerThread
from db.Mysql import Mysql

# SERVER SETTING
hostip = '172.30.1.33'
port = 5882
# SETTING END

#SOCKET INITIALIZE
try:
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind((hostip, port))
    soc.listen()
    print(hostip+":"+str(port)+" IS READY !!")
except:
    print("SOCKET OPEN ERROR")
    exit(1)

Mysql.instance()
st = ServerThread.ServerThread()

while True:
    print("SOCKET IS WAITING FOR ACCEPTING..")
    conn, addr = soc.accept()
    c = Server.Server(conn, addr).start()
    st.append(st)

soc.close()