from db.Mysql import Mysql
import re

"""
0 - 분식
1 - 한식
2 - 경양식
3 - 호프/통닭
4 - 기타
5 - 일식
6 - 정종/대포집/소주방
7 - 통닭(치킨)
8 - 중국식
9 - 뷔페식
10 - 김밥(도시락)
11 - 패스트푸드
12 - 탕류(보신용)
13 - 외국음식전문점(인도,태국등)
14 - 패밀리레스토랑
15 - 식육(숯불구이)
16 - 회집
17 - 복어취급
18 - 냉면집
19 - 일반조리판매
20 - 이동조리
21 - 제과점영업
"""

class Recommandation():
    def requestRecommandation(self, userkey, data):
        result = list()
        tmp = list()
        r_type = self.getRestaurantType(data)
        u_type = self.getUserType(userkey)
        
        for r_item in r_type:
            tmp.append([r_item[0], u_type[r_item[1]]])

        tmp = sorted(tmp, key=lambda l:l[1], reverse=True)

        for i in tmp:
            result.append(i[0])

        return result

    def getUserType(self, userkey):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0]
        #userkey = Mysql.getInstance().execute("select `key` from `accounts` where `id` = '" + str(userid) + "';")
        datas = Mysql.getInstance().execute("select `category` from paymentdata where `userkey` = "+ str(userkey) + ";")
        for data in datas:
            result[int(data[0])] += 1

        return result

    def getRestaurantType(self, data):
        result = list()
        for item in data:
            matchitem = self.removeBranch(item)

            category = Mysql.getInstance().execute("select `category`, `name` from `restaurant` where `name` LIKE '"+ str(matchitem) + "%';")
            if category:
                result.append([item, int(category[0][0])])
            else:
                print(item + "가 존재하지 않습니다.")
                result.append([item, -1])

        return result

    def addUserTypeData(self, userkey, data): 
        for item in data:
            item = self.removeBranch(item)
            category = Mysql.getInstance().execute("SELECT `category` from `restaurant` where `name` LIKE '"+ str(item) +"%';")
            if category:
                Mysql.getInstance().execute("INSERT INTO `paymentdata` VALUES("+ str(userkey) +", '"+ str(item) +"', "+ str(category[0][0]) +");")


    def removeBranch(self, string):
        rule = re.compile(r"\s[\w]+점$")
        matched = rule.search(string)
        if matched:
            matchitem = string.replace(matched.group(0), "")
        else:
            matchitem = string

        return matchitem