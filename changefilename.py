#import sys
#from os import rename, listdir

d = '\supervisely_person\'  # 파일 위치
files = listdir(d)
# files.remove('이름 안바꿔도 되는 파일')

count = 0
for name in files:
  new_name = "{0:04d}.jpg".format(count)
  rename(name,new_name)
  count += 1