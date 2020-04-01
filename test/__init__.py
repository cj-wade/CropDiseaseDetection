n = input()
global count
count = 1


def inner():
    # 说明使用的count变量为全局的不是局部的
    global count
    print(count)
    count = 5
    print(count)


inner()
print(count)
