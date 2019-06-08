import socket

serverip = 'localhost'
port = 5882

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((serverip, port))

sbuff = bytes("1|dusxo123s|test123", encoding='euc-kr')
sock.send(sbuff)

rbuff = sock.recv(1024)
mm = str(rbuff, encoding='euc-kr')
print('수신 : {0}'.format(mm))

sbuff = bytes("2|testid|3123", encoding='euc-kr')
sock.send(sbuff)
 
rbuff = sock.recv(1024)
mm = str(rbuff, encoding='euc-kr')
print('수신 : {0}'.format(mm))

sbuff = bytes("3|이삭토스트 사가정역점|홍콩음식점|도니버거 장안점|체부동잔치집|롯데리아 까치산역점", encoding='euc-kr')
sock.send(sbuff)
 
rbuff = sock.recv(1024)
mm = str(rbuff, encoding='euc-kr')
header = mm.split("|", 1)
mm = mm.split("|")
for m in mm:
    print('수신 : {0}'.format(m))