from tensorboard.backend.event_processing import event_accumulator

results_0513 = ['./tb_log/ckp_0513_complex/', './tb_log/ckp_0513_optica1/', './tb_log/ckp_0513_optica2/']
results_simu = ['./tb_log/ckp_simu5_complex/', './tb_log/ckp_simu7_complex/', './tb_log/ckp_simu9_complex/', \
    './tb_log/ckp_simu11_complex/', './tb_log/ckp_simu13_complex/']

import matplotlib.pyplot as plt
# 设置西文字体为新罗马字体
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "font.size": 80,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)


fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(111)
ax1.set_xlim(0,255)




#加载并保存日志数据
for tb_log in results_0513:
    log = event_accumulator.EventAccumulator(tb_log)
    log.Reload()
    l1_loss = log.scalars.Items('valid_L1loss')
    if tb_log.split('_')[-1][:-1] == 'optica1':
        ax1.plot([i.step for i in l1_loss],[i.value for i in l1_loss],label='UNet')
    elif tb_log.split('_')[-1][:-1] == 'optica2':
        ax1.plot([i.step for i in l1_loss],[i.value for i in l1_loss],label='IDiffNet')
    elif tb_log.split('_')[-1][:-1] == 'complex':
        ax1.plot([i.step for i in l1_loss],[i.value for i in l1_loss],label='Ours')
    else:
        ax1.plot([i.step for i in l1_loss],[i.value for i in l1_loss],label=tb_log.split('_')[-1][:-1])

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Valid_L1loss")
plt.legend(loc='upper right')
plt.show()
plt.savefig("0513_l1_loss")

fig_simu=plt.figure(figsize=(6,4))
ax1=fig_simu.add_subplot(111)
ax1.set_xlim(0,255)

#加载并保存日志数据
for tb_log in results_simu:
    log = event_accumulator.EventAccumulator(tb_log)
    log.Reload()
    l1_loss = log.scalars.Items('valid_L1loss')
    ax1.plot([i.step for i in l1_loss],[i.value for i in l1_loss],label=tb_log.split('_')[-2])

ax1.set_xlabel("epoch")
ax1.set_ylabel("valid_L1loss")
plt.legend(loc='upper right')
plt.show()
plt.savefig("simu_l1_loss")
