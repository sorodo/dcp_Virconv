import random

def generate_random_numbers(max_num):
    if max_num < 12:
        raise ValueError("max_num 必须大于等于 12")

    random_numbers = random.sample(range(1, max_num + 1), 12)
    return random_numbers

# 示例用法
if __name__ == "__main__":
    max_num = 40
    random_numbers = generate_random_numbers(max_num)
    random_numbers.sort()
    print(random_numbers)
    print('hard:',random.sample(range(1, 15),1))
