import enum

@enum.unique
class ReceivePacketHeader(enum.Enum):
    LoginRequest = 1
    RegistAccount = 2
    RequestListSort = 3
    

@enum.unique
class SendPacketHeader(enum.Enum):
    LoginOK = 1
    LoginNO = 2
    RegistOK = 3
    RegistNO = 4
    ListSortOK = 5