
class SingletonInstance():
    """description of class"""
    __instance = None

    @classmethod
    def getInstance(cls):
        if SingletonInstance.__instance == None:
            SingletonInstance()
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.getInstance 
        return cls.__instance