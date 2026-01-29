import numpy as np
import math


def sym_func_p(list, num):
    # creates sublists from main list

    sub_lists = np.array_split(list, 4)

    # prints sublists
    print(sub_lists[0])
    print(sub_lists[1])
    print(sub_lists[2])
    print(sub_lists[3])


    # finds top numers in sublists N

    N = num
    area_1_top = sub_lists[0][-N:]
    area_2_top = sub_lists[1][-N:]
    area_3_top = sub_lists[2][-N:]
    area_4_top = sub_lists[3][-N:]

    print("top  numbers")
    for i in range(1, 5):
        print(eval("area_%d_top" % (i)))


    # finds median value of the top numbers  in sublists
    area_1_topM = np.median(area_1_top)
    area_2_topM = np.median(area_2_top)
    area_3_topM = np.median(area_3_top)
    area_4_topM = np.median(area_4_top)


    # top symettry

    first_matchT = math.isclose(area_1_topM, area_3_topM, abs_tol= 0.02)
    second_matchT = math.isclose(area_1_topM, area_4_topM, abs_tol= 0.02)
    third_matchT = math.isclose(area_2_topM, area_3_topM, abs_tol=0.02)
    fourth_matchT = math.isclose(area_2_topM, area_4_topM, abs_tol=0.02)

    if area_1_topM > 1.05:
        if first_matchT == True or second_matchT == True:
            result = "symmetry"
            print(result)
    elif area_2_topM > 1.05:
        if third_matchT == True or fourth_matchT == True:
            result = "symmetry"
            print(result)
    else:
        result = "no symmetry"
        print(result)

    return result




def sym_func_t(list, num):
    # creates sublists from main list

    sub_lists = np.array_split(list, 4)

    # prints sublists
    print(sub_lists[0])
    print(sub_lists[1])
    print(sub_lists[2])
    print(sub_lists[3])


    # finds top numers in sublists N

    N = num
    # finds bottom two numers in sublists
    area_1_bot = sub_lists[0][:N]
    area_2_bot = sub_lists[1][:N]
    area_3_bot = sub_lists[2][:N]
    area_4_bot = sub_lists[3][:N]

    print("bottom numbers")
    for i in range(1, 5):
        print(eval("area_%d_bot" % (i)))

    # finds median value of the twop 2 numbers and bottom two numers in sublists

    area_1_botM = np.median(area_1_bot)
    area_2_botM = np.median(area_2_bot)
    area_3_botM = np.median(area_3_bot)
    area_4_botM = np.median(area_4_bot)



    first_matchB = math.isclose(area_1_botM, area_3_botM, abs_tol=0.02)
    second_matchB = math.isclose(area_2_botM, area_3_botM, abs_tol=0.02)
    third_matchB = math.isclose(area_1_botM, area_4_botM, abs_tol=0.02)
    fourth_matchB = math.isclose(area_2_botM, area_4_botM, abs_tol=0.02)

    if area_1_botM < 0.95:
        if first_matchB == True or second_matchB == True:
            result = "symmetry"
            print(result)
    elif area_2_botM < 0.95:
        if third_matchB == True or fourth_matchB == True:
            result = "symmetry"
            print(result)
    else:
        result = "no symmetry"
        print(result)

    return result
