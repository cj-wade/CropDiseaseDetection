# 物种种类
def crop_class(id_class):
    # 苹果
    if 0 <= id_class < 6:
        if id_class == 0:
            return 0, 0
        elif 1 <= id_class <= 2:
            return 0, 1
        elif id_class == 3:
            return 0, 2
        else:
            return 0, 3

    # 樱桃
    if 6 <= id_class < 9:
        if id_class == 6:
            return 1, 4
        else:
            return 1, 5

    # 玉米
    if 9 <= id_class < 17:
        if id_class == 9:
            return 2, 6
        elif 10 <= id_class <= 11:
            return 2, 7
        elif 12 <= id_class <= 13:
            return 2, 8
        elif 14 <= id_class <= 15:
            return 2, 9
        else:
            return 2, 10

    # 葡萄
    if 17 <= id_class < 24:
        if id_class == 17:
            return 3, 11
        elif 18 <= id_class <= 19:
            return 3, 12
        elif 20 <= id_class <= 21:
            return 3, 13
        else:
            return 3, 14

    # 柑桔
    if 24 <= id_class < 27:
        if id_class == 24:
            return 4, 15
        else:
            return 4, 16

    # 桃
    if 27 <= id_class < 30:
        if id_class == 27:
            return 5, 17
        else:
            return 5, 18

    # 辣椒
    if 30 <= id_class < 33:
        if id_class == 30:
            return 6, 19
        else:
            return 6, 20

    # 马铃薯
    if 33 <= id_class < 37:
        if id_class == 33:
            return 7, 21
        elif 34 <= id_class <= 35:
            return 7, 22
        else:
            return 7, 23

    # 草莓
    if 37 <= id_class < 41:
        if id_class == 37:
            return 8, 24
        else:
            return 8, 25

    if 41 <= id_class < 61:
        if id_class == 41:
            return 9, 26
        elif 42 <= id_class <= 43:
            return 9, 27
        elif 44 <= id_class <= 45:
            return 9, 28
        elif 46 <= id_class <= 47:
            return 9, 29
        elif 48 <= id_class <= 49:
            return 9, 30
        elif 50 <= id_class <= 51:
            return 9, 31
        elif 52 <= id_class <= 53:
            return 9, 32
        elif 54 <= id_class <= 55:
            return 9, 33
        elif 56 <= id_class <= 57:
            return 9, 34
        elif 58 <= id_class <= 59:
            return 9, 35
        else:
            return 9, 36
