from datetime import datetime
print(# 尝试将图像帧戳除以 1e7 看看
    datetime.utcfromtimestamp(16233043764692187 / 1e7))
print(16233043764692187 / 1e7)

timestamp = 1631530090129454900 / 1e9
timestamp2 = 1623305186298720600 / 1e9
             #1623305186298720600
# 1631530090129454900
# 2021-09-13 10:44:59.683124
dt = datetime.utcfromtimestamp(timestamp)
print(dt)
print(datetime.utcfromtimestamp(timestamp2))

