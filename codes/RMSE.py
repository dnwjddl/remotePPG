import os
import numpy as np


#RMSE 값 도출
def rmse(pred, target):
    return np.sqrt(((pred - target)**2).mean())


#@pred_avg와 @gt_avg 간의 rmse 도출
def Compare_avg(lines):
    try:
        pred_avg = [int(x) for x in lines[1].strip().split(',')]
        gt_avg = [int(x) for x in lines[3].strip().split(',')]
        return rmse(np.array(pred_avg), np.array(gt_avg))
    except ValueError:
        return 'NULL'


#@pred_avg와 @gt_inst 간의 rmse 도출
def Compare_inst(lines):
    try:
        pred_avg = [int(x) for x in lines[1].strip().split(',')]
        gt_inst = [int(x) for x in lines[5].strip().split(',')]
        return rmse(np.array(pred_avg), np.array(gt_inst))
    except ValueError:
        return 'NULL'

def main():
    dir_path = 'C:/Users/woojung/Desktop/AI_WORKS/GIT/loc-git/rPPG/최종 코드/PURE'
    dir_list = os.listdir(dir_path)
    print('Directory List:\n' + str(dir_list))

    output = open('./result.txt', 'w')
    print("open")

    for file in dir_list:   #디렉토리 접근
        print('Dir name: ' + file)
        output.write(file + '\n')

        file_name = dir_path + '/' + file
        txt_list = os.listdir(file_name)

        for txt in txt_list:    #디렉토리 내 파일 접근
            txt_name = file_name + '/' + txt

            with open(txt_name) as f:   #파일 값 읽어온 후 계산 & 결과 출력
                lines = f.readlines()
                result = ('%-30s%-20s%-20s' %(txt, Compare_avg(lines), Compare_inst(lines)))
                print(result)
                output.write(result + '\n')

if __name__ == '__main__':
    main()

