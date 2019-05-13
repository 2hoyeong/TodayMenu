import pymysql
from design.SingletonPattern import SingletonInstance

class Mysql(SingletonInstance):
    def __init__(self):
        self.connection = pymysql.connect(host='localhost', user='root', password='apmsetup', db='food', charset='utf8')
        self.cursor = self.connection.cursor()

    def execute(self, query):
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        self.connection.commit()
        return rows

    #def __del__(self):
    #    connection.close()


