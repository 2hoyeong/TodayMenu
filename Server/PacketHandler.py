from Header import ReceivePacketHeader
from Header import SendPacketHeader
from db.Mysql import Mysql
from Recommandation import Recommandation

class PacketHandler:
    def switch(self, conn, header, data):
        self.case_name = str(ReceivePacketHeader(int(header)).name)
        self.case = getattr(self, self.case_name)(conn, data)

    def LoginRequest(self, conn, data):
        id, pw = data.split("|", 1)
        rows = Mysql.getInstance().execute("select * from `accounts` where `id` = '"+ id + "' and `pw` = '" + pw + "'")
        if rows:
            conn.send(SendPacketHeader.LoginOK, "로그인이 완료되었습니다")
            conn.id = id
            conn.key = int(rows[0][0])
        else:
            conn.send(SendPacketHeader.LoginNO, "로그인에 실패했습니다")


    def RegistAccount(self, conn, data):
        id, pw = data.split("|", 1)
        rows = Mysql.getInstance().execute("select * from `accounts` where `id` = '"+ id + "'")
        if rows:
            conn.send(SendPacketHeader.RegistNO, "중복된 아이디 입니다!")
        else:
            Mysql.getInstance().execute("INSERT INTO accounts VALUES(0, '"+ id +"', '"+ pw +"')")
            conn.send(SendPacketHeader.RegistOK, "회원가입이 완료되었습니다!")

    def RequestListSort(self, conn, data):
        if not conn.id:
            print("비정상적인 접근 : " + data)
            return

        Restaurant = list()
        for item in data.split("|"):
            Restaurant.append(item)
        if Restaurant:
            result = Recommandation().requestRecommandation(conn.key, Restaurant)
            conn.send(SendPacketHeader.ListSortOK, '|'.join(result))

    def AddPaymentData(self, conn, data):
        Restaurant = list()
        for item in data.split("|"):
            Restaurant.append(item)
        if Restaurant:
            Recommandation().addUserTypeData(conn.key, Restaurant)