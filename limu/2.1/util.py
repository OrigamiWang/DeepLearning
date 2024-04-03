cnt = 0


def p(*values: object):
    global cnt
    cnt += 1
    idx = f'NO.{cnt}: '
    print(idx, *values)
    print()
