from my_eval.my_evalutor import my_evalutor

# --gt
#     --DUTLF
#     --HFUT
# --pred
#     --JLDCF
#         --DUTLF
#         --HFUT



my_evalutor(save_dir = './', gt_dir = 'D:\work\models\Evaluate-SOD\gt',
            pred_dir = 'D:\work\models\Evaluate-SOD\pred', cuda = True)