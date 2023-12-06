from utils import acc_cal
import cv2

img_name = ["0618.png", "0854.png", "1066.png"]
img_name = img_name[0]
output_dir = "output_auto_sample_smo"
acc_output_dir = "acc_result"
# 存放acc的列表
acc_list = []
img_mask = cv2.imread("data/" + img_name[:-4] + "_mask" + img_name[-4:], flags=0)

for itv in [x for x in range(15, 55, 5)]:
    img = cv2.imread(output_dir + "/" + img_name[:-4] + f"_{itv}/" + img_name[:-4] + "_res" + img_name[-4:], flags=0)
    acc = acc_cal(img, img_mask)
    # 写成百分数，并保留两位小数，存入数组acc_list
    acc_list.append('%.2f%%' % (acc * 100))
    print(f'itv: {itv}, acc: {acc_list[-1]}')

# 保存acc列表，保存为txt文件，文件名为acc_result + "/" + output_dir + img_name[:-4] + "_acc.txt"
with open(acc_output_dir + "/" + output_dir + "_" + img_name[:-4] + ".txt", "w") as f:
    for acc in acc_list:
        f.write(str(acc) + "\n")
