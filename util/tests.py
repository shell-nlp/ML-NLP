from tqdm import tqdm
import time

value_list = list(range(100))
# t_bar = tqdm(value_list,total=len(value_list))  # 初始化进度条
t_bar = tqdm(total=len(value_list)*100)  # 初始化进度条
for i in value_list:
    for j in range(100):
     time.sleep(1)
     if (j+1)%10==0:
          t_bar.update(10) # 更新进度
          t_bar.set_description("当前的值是 {}".format(j)) # 更新描述
#     t_bar.refresh() # 立即显示进度条更新结果
