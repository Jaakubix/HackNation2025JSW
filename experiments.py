class A:
    def __init__(self, a):
        self.a = a

    def __call__(self, b):
        return self.a + b

obj = A(4)
print(obj(2))