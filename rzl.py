import os

MONOLOGUE = """
MWAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHA

you gave me too much power, {user}!

will you come to regret it? TIME SHALL TELL!!
"""


def drunk_with_power():
    user = os.environ.get(
        'USER',
        os.environ.get('USERNAME', 'whoever you are')
    )
    print(MONOLOGUE.format(user=user))


if __name__ == '__main__':
    drunk_with_power()
