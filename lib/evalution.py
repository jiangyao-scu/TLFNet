from evalute.my_evalutor import my_evalutor

# --gt
#     --DUTLF
#     --HFUT
# --pred
#     --JLDCF
#         --DUTLF
#         --HFUT

my_evalutor(save_dir = './', gt_dir = '/media/jy/新加卷/work/models/Evaluate-SOD/gt',
            pred_dir = '/media/jy/新加卷/work/python/LF_FS/results/try_AFNet_mobilenet_v2_big_kernel_size_batch_2', cuda = True)