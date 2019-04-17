from Header import ReceivePacketHeader
from db.Mysql import Mysql

class PacketHandler:
    def switch(self, conn, header, data):
        self.case_name = str(ReceivePacketHeader(int(header)).name)
        self.case = getattr(self, self.case_name)(conn, data)

    def LoginRequest(self, conn, data):
        id, pw = data.split("|", 1)
        #print(query)
        rows = Mysql.getInstance().execute("select * from `accounts` where `id` = '"+ id + "' and `pw` = '" + pw + "'")
        print(rows) 
        conn.send(2, "로그인이 완료되었습니다!")


    def RegistAccount(self, conn, data):
        print("REGIST")