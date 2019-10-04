import datetime

def hello_world():
    print('Hello world!')
    now = datetime.datetime.now()
    print('It is {}'.format(now))

if __name__=="__main__":
    hello_world()
