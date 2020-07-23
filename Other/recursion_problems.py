import random
import time


def log_2(n):
    # round down log_2
    n = int(n)
    if n == 1:
        return 0
    else:
        return 1 + log_2(n/2)


def cryptography(password):
    if not isinstance(password, str):
        password = str(password)

    asc = 1
    for letter in password:
        asc *= ord(letter)**2

    asc = str(asc)
    asc = asc[(len(asc)-1) // 2 - len(password): (len(asc)-1) // 2 + len(password)]

    result = ''
    for idx in range(2, len(asc)+2, 2):
        result += chr(int(asc[idx-2:idx])+30)

    return result


def guess_roll(n):
    assert isinstance(n, int) and 0 < n < 7, 'wrong input'
    roll = random.randint(1, 6)
    print('the roll is', end='')
    for i in range(3):
        time.sleep(.5)
        print('.', end='')

    if roll == n:
        print(f' {roll}! CORRECT')
    else:
        print(f' {roll}! WRONG')


def ascii_to_letter():
    return [(i, chr(i)) for i in range(33, 200)]



if __name__ == '__main__':
    my_pass = 'luni puppy wants to go out!'
    print(cryptography(my_pass))
    # guess_roll(1)
    # print(ascii_to_letter())