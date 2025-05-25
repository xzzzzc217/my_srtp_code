# my_srtp_code
本仓库用于托管本人国家级srtp的相关代码

其中ci_grok3是做信道无关处理（channel independent 简称 ci）后得到频谱图后（如下图所示）。将图片传入cnn模型进行设备识别
![ci_exam](https://github.com/user-attachments/assets/a0babffa-058c-4514-93ea-0c5e06e30de1) ![roman前导码](https://github.com/user-attachments/assets/0e56fa2e-50be-4ff8-ade8-a1565576f533)

得到设备分类混淆矩阵如下图所示：

![ci_tensor_ttod](https://github.com/user-attachments/assets/d94c5542-96f8-47cd-9fa9-75b20a4c4a2b) ![ci_tensor_ttsd](https://github.com/user-attachments/assets/27c57dbb-8dc5-4ac8-92ce-75b5aca9fb26)


preprocessing中覆盖了自24年11月到现在（25年5月）的所有预处理程序

pre_v4是最新的预处理流程，包括帧同步、前导码提取、cfo补偿，但是特征提取函数没有声明（目前还没确定）

pre_claud2能较好地提取相位差分特征并可视化：
![phase_noise_diff_features_visualization_27](https://github.com/user-attachments/assets/60dd6a8e-0620-4a7a-a0c3-bb2c859c9694)
![phase_noise_diff_features_visualization_29](https://github.com/user-attachments/assets/0fdca518-10dc-4308-87f5-adb68b34b8dc)
![phase_noise_diff_features_visualization_32](https://github.com/user-attachments/assets/4ce03ff2-6ef2-4acb-9736-92a6bbc16003)
![phase_noise_diff_features_visualization_35](https://github.com/user-attachments/assets/6d911d26-07dd-4175-870d-8b26024381c0)
pre_gemini3尝试提取方差、平均值、峰度等特征：
![diff_mean](https://github.com/user-attachments/assets/0a608a92-b7ae-451d-b60a-9bac36992ee8) ![1_mean](https://github.com/user-attachments/assets/922dcaf2-61d2-4942-b1ce-f29fe0f05490)
