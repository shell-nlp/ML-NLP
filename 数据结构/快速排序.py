def quick_sort(num: list):
    if len(num) <= 1:
        return num
    p = num[0]
    left = [x for x in num[1:] if x < p]
    right = [x for x in num[1:] if x > p]
    return quick_sort(left) + [p] + quick_sort(right)


if __name__ == '__main__':
    num = [5, 9, 3, 4, 7, 6, 2, 6, -1]
    print(quick_sort(num))
