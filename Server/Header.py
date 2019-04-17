import enum

@enum.unique
class ReceivePacketHeader(enum.Enum):
    LoginRequest = 0x01
    RegistAccount = 0x02
    

@enum.unique
class SendPacketHeader(enum.Enum):
    LoginOK = 0x01
    LoginNO = 0x02
    RegistOK = 0x03
    RegistNO = 0x04