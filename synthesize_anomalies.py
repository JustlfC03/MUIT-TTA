import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label, gaussian_filter
import skimage.exposure


def keep_largest_connected_component_2d_0(image_2d_0):
    """保留函数定义，避免引用报错，实际无功能"""
    return image_2d_0


def create_circular_kernel_0(radius_0):
    """保留函数定义，避免引用报错，实际无功能"""
    pass


def dilate_segmentation_2d_0(segmentation_array_0):
    """保留函数定义，避免引用报错，实际返回空掩码"""
    return np.zeros_like(segmentation_array_0, dtype=np.uint8)


def generate_anomaly_mask_2d_0(dilate_mask_0, original_mask_0, height_0, width_0, min_size_0, max_size_0):
    """保留函数定义，返回全0掩码"""
    return np.zeros((height_0, width_0), dtype=np.uint8)


def generate_stripes_mask_0(shape_0, anomaly_mask_0, density_0=0.12, width_range_0=(2, 5)):
    """保留函数定义，返回全0条纹掩码"""
    return np.zeros(shape_0, dtype=np.uint8)


def add_anomaly_to_image_0(image_0, anomaly_mask_0, original_mask_0):
    """移除所有出血生成逻辑，仅保留原始图像+背景置0"""
    # 直接返回原始图像，仅将掩码为0的区域设为黑色（与原代码背景逻辑一致）
    result = image_0.copy()
    result[original_mask_0 == 0] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def check_valid_pixel_ratio(image_2d, threshold=0.2):
    """
    检查图像有效像素占比（像素值不为0的部分）
    :param image_2d: 二维图像数组（最终要保存的图像）
    :param threshold: 有效占比阈值（默认30%）
    :return: 占比是否达到阈值，具体占比
    """
    total_pixels = image_2d.size  # 总像素数
    valid_pixels = np.count_nonzero(image_2d)  # 非0像素数（有效像素）
    valid_ratio = valid_pixels / total_pixels  # 有效占比
    return valid_ratio >= threshold, valid_ratio  # 返回是否达标和具体占比


def process_single_image_0(data_path_0, mask_path_0, output_data_dir_0, output_mask_dir_0, min_mask_size_0,
                           max_mask_size_0):
    os.makedirs(output_data_dir_0, exist_ok=True)
    os.makedirs(output_mask_dir_0, exist_ok=True)

    try:
        with Image.open(data_path_0) as img_0:
            image_0 = np.array(img_0.convert('L'), dtype=np.float32)
        # 修复：将上下文变量名改为mask_img，避免与mask_path变量冲突
        with Image.open(mask_path_0) as mask_img_0:
            original_mask_0 = np.array(mask_img_0.convert('L'), dtype=np.uint8)

        height_0, width_0 = image_0.shape
        print(f"处理 {os.path.basename(data_path_0)} ({height_0}x{width_0})")

        # 生成全0异常掩码（无实际出血）
        anomaly_mask_0 = np.zeros_like(original_mask_0, dtype=np.uint8)

        # 调用处理函数（仅背景置0，无出血）- 得到初步处理后的图像
        hemo_image_0 = add_anomaly_to_image_0(image_0=image_0, anomaly_mask_0=anomaly_mask_0,
                                              original_mask_0=original_mask_0)

        # 确保保存前掩码为0的区域已被清除（最终要保存的图像）
        final_image_0 = hemo_image_0 * (original_mask_0 > 0).astype(np.uint8)

        # 核心修改：对最终要保存的图像进行有效像素占比判断
        valid_flag, valid_ratio = check_valid_pixel_ratio(final_image_0, 0.3)
        if not valid_flag:
            print(f"跳过 {os.path.basename(data_path_0)}：最终保存图像的有效像素占比{valid_ratio:.2%} < 30%")
            return

        # 核心修改：在文件名前添加"0_"前缀（保留原命名逻辑）
        base_name_0 = os.path.basename(data_path_0)
        new_base_name_0 = f"0_{base_name_0}"  # 拼接前缀

        # 保存处理后的图片（带0_前缀，为最终处理后的图像）
        Image.fromarray(final_image_0).save(os.path.join(output_data_dir_0, new_base_name_0))
        # 保存掩码图片（带0_前缀，全0掩码）
        Image.fromarray((anomaly_mask_0 * 255).astype(np.uint8)).save(os.path.join(output_mask_dir_0, new_base_name_0))
        print(f"已保存 {new_base_name_0}（有效像素占比{valid_ratio:.2%}，掩码大小: {np.sum(anomaly_mask_0)}像素）")

    except Exception as e_0:
        print(f"处理 {data_path_0} 失败: {e_0}")


def process_all_images_0(data_dir_0, mask_dir_0, output_data_dir_0, output_mask_dir_0, min_mask_size_0,
                         max_mask_size_0):
    for file_0 in [f_0 for f_0 in os.listdir(data_dir_0) if f_0.lower().endswith('.png')]:
        data_path_0 = os.path.join(data_dir_0, file_0)
        mask_path_0 = os.path.join(mask_dir_0, file_0)
        if os.path.exists(mask_path_0):
            process_single_image_0(
                data_path_0=data_path_0,
                mask_path_0=mask_path_0,
                output_data_dir_0=output_data_dir_0,
                output_mask_dir_0=output_mask_dir_0,
                min_mask_size_0=min_mask_size_0,
                max_mask_size_0=max_mask_size_0
            )
        else:
            print(f"跳过 {file_0}：掩码文件不存在")


import skimage.exposure


def keep_largest_connected_component_2d_1(image_2d_1):
    if not np.any(image_2d_1):
        return image_2d_1

    labeled_1, num_labels_1 = label(image_2d_1)
    if num_labels_1 == 0:
        return image_2d_1

    component_sizes_1 = np.bincount(labeled_1.flatten())
    component_sizes_1[0] = 0
    max_label_1 = np.argmax(component_sizes_1)

    largest_component_1 = np.zeros_like(image_2d_1, dtype=image_2d_1.dtype)
    largest_component_1[labeled_1 == max_label_1] = image_2d_1[labeled_1 == max_label_1]
    return largest_component_1


def create_circular_kernel_1(radius_1):
    diameter_1 = 2 * radius_1 + 1
    kernel_1 = np.zeros((diameter_1, diameter_1), dtype=np.uint8)
    center_1 = (radius_1, radius_1)
    cv2.circle(kernel_1, center_1, radius_1, 1, -1)
    return kernel_1.astype(bool)


def dilate_segmentation_2d_1(segmentation_array_1):
    # 基础区域改为label=1，用于生成异常掩码的范围
    seg_mask_1 = (segmentation_array_1 == 1).astype(bool)
    struct_elem_1 = create_circular_kernel_1(radius_1=15)

    dilated_1 = binary_dilation(seg_mask_1, struct_elem_1, iterations=1).astype(np.uint8)
    # 限制在label=1区域内生成异常掩码的候选范围
    dilated_1[segmentation_array_1 != 1] = 0
    dilated_1 = binary_dilation(dilated_1.astype(bool), struct_elem_1, iterations=1).astype(np.uint8)
    # 确保异常掩码的候选范围在label=1内
    dilated_1 = dilated_1 * (segmentation_array_1 == 1).astype(np.uint8)

    return keep_largest_connected_component_2d_1(dilated_1)


def generate_anomaly_mask_2d_1(dilate_mask_1, original_mask_1, height_1, width_1, min_size_1, max_size_1):
    anomaly_mask_1 = np.zeros((height_1, width_1), dtype=np.uint8)
    loop_num_1 = 0
    max_loops_1 = 50
    threshold_1 = 200

    while True:
        noise_1 = np.random.randint(0, 256, (height_1, width_1), dtype=np.uint8)
        blur_1 = gaussian_filter(noise_1.astype(np.float32), sigma=15)
        stretch_1 = skimage.exposure.rescale_intensity(blur_1, out_range=(0, 255)).astype(np.uint8)
        thresh_1 = (stretch_1 >= threshold_1).astype(np.uint8) * 255

        kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_OPEN, kernel_1)
        closed_1 = cv2.morphologyEx(opened_1, cv2.MORPH_CLOSE, kernel_1)

        # 严格限制异常掩码在label=1区域内
        anomaly_mask_1 = (dilate_mask_1 * closed_1 * (original_mask_1 == 1)).astype(np.uint8)
        anomaly_mask_1 = np.where(anomaly_mask_1 > 0, 1, 0).astype(np.uint8)
        mask_size_1 = np.sum(anomaly_mask_1)

        if min_size_1 <= mask_size_1 <= max_size_1:
            break
        loop_num_1 += 1
        if loop_num_1 >= max_loops_1:
            print(f"警告：达到最大循环次数({max_loops_1})，无法生成合适掩码")
            break

    return anomaly_mask_1


def generate_stripes_mask_1(shape_1, anomaly_mask_1, density_1=0.12, width_range_1=(2, 5)):
    """生成异常区域内的无规则条纹掩码"""
    height_1, width_1 = shape_1
    stripes_1 = np.zeros(shape_1, dtype=np.uint8)

    # 随机生成条纹数量（基于异常区域大小）
    mask_size_1 = np.sum(anomaly_mask_1)
    num_stripes_1 = max(2, int(mask_size_1 * density_1 / 100))  # 降低密度，避免条纹过密

    for _ in range(num_stripes_1):
        # 随机选择条纹起点（必须在异常区域内）
        y_coords_1, x_coords_1 = np.where(anomaly_mask_1 == 1)
        if len(y_coords_1) == 0:
            continue  # 异常区域为空则跳过
        idx_1 = np.random.randint(0, len(y_coords_1))
        y1_1, x1_1 = y_coords_1[idx_1], x_coords_1[idx_1]

        # 随机生成条纹方向和长度
        angle_1 = np.random.uniform(0, np.pi)  # 0-180度随机角度
        length_1 = np.random.uniform(8, min(height_1, width_1) * 0.35)  # 条纹长度

        # 计算条纹终点
        y2_1 = int(y1_1 + length_1 * np.sin(angle_1))
        x2_1 = int(x1_1 + length_1 * np.cos(angle_1))

        # 确保终点在图像范围内
        y2_1 = np.clip(y2_1, 0, height_1 - 1)
        x2_1 = np.clip(x2_1, 0, width_1 - 1)

        # 随机条纹宽度
        stripe_width_1 = np.random.randint(*width_range_1)

        # 绘制条纹（使用抗锯齿线条）
        cv2.line(stripes_1, (x1_1, y1_1), (x2_1, y2_1), 1, stripe_width_1)

    # 确保条纹只在异常区域内（间接限制在label=1）
    stripes_1 = stripes_1 * anomaly_mask_1
    return stripes_1


def add_anomaly_to_image_1(image_1, anomaly_mask_1, original_mask_1):
    """
        仅将anomaly_mask和条纹限制在label=1处，其他膨胀区域不限制。
        外围出血透明度(异常颜色占比)设置为0.8，原图透明度占比0.2。
        """

    # --- 核心修改参数：外围透明度 ---
    # 异常颜色占比 (透明度)
    ANOMALY_ALPHA_1 = 0.8
    # 原图颜色占比
    ORIGINAL_ALPHA_1 = 1.0 - ANOMALY_ALPHA_1
    # -------------------------------

    """仅将anomaly_mask和条纹限制在label=1处，其他膨胀区域不限制"""
    # 严格限制异常掩码在label=1区域
    anomaly_mask_1 = anomaly_mask_1 * (original_mask_1 == 1).astype(np.uint8)

    # 生成条纹掩码（因基于anomaly_mask，故自动限制在label=1）
    stripes_mask_1 = generate_stripes_mask_1(shape_1=image_1.shape, anomaly_mask_1=anomaly_mask_1)

    # 条纹边缘模糊区域（因基于条纹掩码，故自动限制在label=1）
    stripe_blur_kernel_1 = create_circular_kernel_1(radius_1=1)
    stripes_mask_blur_area_1 = binary_dilation(stripes_mask_1.astype(bool), stripe_blur_kernel_1, iterations=3).astype(
        np.uint8)
    stripes_edge_area_1 = stripes_mask_blur_area_1 - stripes_mask_1

    # 仅计算label>0区域的像素均值
    # 生成label>0的掩码
    valid_mask_1 = (original_mask_1 > 0).astype(bool)
    # 提取有效区域的像素
    valid_pixels_1 = image_1[valid_mask_1]
    rand_1 = np.random.uniform(1.0, 1.2)
    # 计算有效区域的均值，若无有效像素则使用默认值50
    data_mean_1 = np.mean(valid_pixels_1) if len(valid_pixels_1) > 0 else 50
    base_offset_1 = data_mean_1 * 1.2 * rand_1

    # 随机噪声缩放比例（用于基础异常区域）
    noise_choice_1 = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
    if noise_choice_1 == 1:
        base_scale_1 = np.random.uniform(0.1, 0.9, anomaly_mask_1.shape)
    elif noise_choice_1 == 2:
        base_scale_1 = np.random.uniform(0.2, 0.8, anomaly_mask_1.shape)
    else:
        base_scale_1 = np.random.uniform(0.3, 0.7, anomaly_mask_1.shape)

    # 计算基础异常区域颜色（仅在label=1内）
    base_anomaly_color_1 = anomaly_mask_1 * (image_1 * base_scale_1 + base_offset_1)

    # 基础异常区域叠加（仅在label=1内修改）
    hemo_image_1 = image_1.copy()
    hemo_image_1[original_mask_1 == 1] = (image_1[original_mask_1 == 1] * (1 - anomaly_mask_1[original_mask_1 == 1]) +
                                          base_anomaly_color_1[original_mask_1 == 1])

    # 第一次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem1_1 = create_circular_kernel_1(radius_1=1)
    dilate1_1 = binary_dilation(anomaly_mask_1.astype(bool), struct_elem1_1, iterations=1).astype(np.uint8)
    dilate1_1 = dilate1_1 * (original_mask_1 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate1_only_1 = dilate1_1 - anomaly_mask_1

    # deep_scale1_1 = dilate1_only_1 * np.random.uniform(0.05, 0.3, dilate1_only_1.shape)
    # deep_offset1_1 = dilate1_only_1 * (data_mean_1 * 1.1 * rand_1)
    # hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1) + (image_1 * deep_scale1_1 + deep_offset1_1)

    # 外围异常颜色计算
    deep_scale1_1 = dilate1_only_1 * np.random.uniform(0.05, 0.3, dilate1_only_1.shape)
    deep_offset1_1 = dilate1_only_1 * (data_mean_1 * 1.1 * rand_1)
    anomaly_color1_1 = (image_1 * deep_scale1_1 + deep_offset1_1)

    # --- 第一次膨胀区域：应用 0.8 / 0.2 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1) + (image_1 * deep_scale1_1 + deep_offset1_1)
    # 修改为：新图像 = 旧图像 * (1 - 0.6*Mask) + 异常颜色 * 0.6*Mask
    hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1 * ANOMALY_ALPHA_1) + anomaly_color1_1 * ANOMALY_ALPHA_1

    # 第二次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem2_1 = create_circular_kernel_1(radius_1=1)
    dilate2_1 = binary_dilation(dilate1_1.astype(bool), struct_elem2_1, iterations=2).astype(np.uint8)
    dilate2_1 = dilate2_1 * (original_mask_1 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate2_only_1 = dilate2_1 - dilate1_1

    # deep_scale2_1 = np.random.uniform(0.02, 0.2, image_1.shape)
    # deep_offset2_1 = data_mean_1 * 0.9
    # hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1) + (
    #         image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))

    # 外围异常颜色计算
    deep_scale2_1 = np.random.uniform(0.02, 0.2, image_1.shape)
    deep_offset2_1 = data_mean_1 * 0.9
    anomaly_color2_1 = (image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))

    # --- 第二次膨胀区域：应用 0.8 / 0.2 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1) + (image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))
    # 修改为：新图像 = 旧图像 * (1 - 0.8*Mask) + 异常颜色 * 0.8*Mask
    hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1 * ANOMALY_ALPHA_1) + anomaly_color2_1 * ANOMALY_ALPHA_1

    # 条纹叠加（仅在label=1内）
    stripes_base_color_1 = base_anomaly_color_1 * stripes_mask_1
    stripes_color_1 = stripes_base_color_1 * 1.2
    hemo_image_1 = hemo_image_1 * (1 - stripes_mask_1) + stripes_color_1

    # 条纹边缘模糊处理（仅在label=1内）
    if np.sum(stripes_edge_area_1) > 0:
        edge_blur_1 = gaussian_filter(hemo_image_1, sigma=1.0)
        hemo_image_1 = hemo_image_1 * (1 - stripes_edge_area_1 * 1.2) + edge_blur_1 * (stripes_edge_area_1 * 1.2)

    # 将掩码为0的区域设为0（黑色背景）
    hemo_image_1[original_mask_1 == 0] = 0

    return np.clip(hemo_image_1, 0, 255).astype(np.uint8)


def process_single_image_1(data_path_1, mask_path_1, output_data_dir_1, output_mask_dir_1, min_mask_size_1,
                           max_mask_size_1):
    os.makedirs(output_data_dir_1, exist_ok=True)
    os.makedirs(output_mask_dir_1, exist_ok=True)

    try:
        with Image.open(data_path_1) as img_1:
            image_1 = np.array(img_1.convert('L'), dtype=np.float32)
        # 修复：将上下文变量名改为mask_img，避免与mask_path变量冲突
        with Image.open(mask_path_1) as mask_img_1:
            original_mask_1 = np.array(mask_img_1.convert('L'), dtype=np.uint8)

        # 检查是否存在label=1的区域（因异常掩码需在此生成）
        if not np.any(original_mask_1 == 1):
            print(f"跳过 {os.path.basename(data_path_1)}：无值为1的掩码区域")
            return

        height_1, width_1 = image_1.shape
        print(f"处理 {os.path.basename(data_path_1)} ({height_1}x{width_1})")

        hemo_range_1 = dilate_segmentation_2d_1(segmentation_array_1=original_mask_1)
        if np.sum(hemo_range_1) < min_mask_size_1:
            print(f"跳过 {os.path.basename(data_path_1)}：label=1区域内可生成异常的范围过小")
            return

        anomaly_mask_1 = generate_anomaly_mask_2d_1(
            dilate_mask_1=hemo_range_1,
            original_mask_1=original_mask_1,
            height_1=height_1,
            width_1=width_1,
            min_size_1=min_mask_size_1,
            max_size_1=max_mask_size_1
        )
        mask_size_1 = np.sum(anomaly_mask_1)
        if not (min_mask_size_1 <= mask_size_1 <= max_mask_size_1):
            print(f"跳过 {os.path.basename(data_path_1)}：掩码大小不符 ({mask_size_1}像素)")
            return

        hemo_image_1 = add_anomaly_to_image_1(image_1=image_1, anomaly_mask_1=anomaly_mask_1,
                                              original_mask_1=original_mask_1)

        # 确保保存前掩码为0的区域已被清除
        hemo_image_1 = hemo_image_1 * (original_mask_1 > 0).astype(np.uint8)

        # 核心修改：对最终要保存的图像进行有效像素占比判断
        valid_flag, valid_ratio = check_valid_pixel_ratio(hemo_image_1)
        if not valid_flag:
            print(f"跳过 {os.path.basename(data_path_1)}：最终保存图像的有效像素占比{valid_ratio:.2%} < 30%")
            return

        # 核心修改：在文件名前添加"1_"前缀
        base_name_1 = os.path.basename(data_path_1)
        new_base_name_1 = f"1_{base_name_1}"  # 拼接前缀

        # 保存处理后的图片（带1_前缀）
        Image.fromarray(hemo_image_1).save(os.path.join(output_data_dir_1, new_base_name_1))
        # 保存掩码图片（带1_前缀）
        Image.fromarray((anomaly_mask_1 * 255).astype(np.uint8)).save(os.path.join(output_mask_dir_1, new_base_name_1))
        print(f"已保存 {new_base_name_1}（掩码大小: {mask_size_1}像素）")

    except Exception as e_1:
        print(f"处理 {data_path_1} 失败: {e_1}")


def process_all_images_1(data_dir_1, mask_dir_1, output_data_dir_1, output_mask_dir_1, min_mask_size_1,
                         max_mask_size_1):
    for file_1 in [f_1 for f_1 in os.listdir(data_dir_1) if f_1.lower().endswith('.png')]:
        data_path_1 = os.path.join(data_dir_1, file_1)
        mask_path_1 = os.path.join(mask_dir_1, file_1)
        if os.path.exists(mask_path_1):
            process_single_image_1(
                data_path_1=data_path_1,
                mask_path_1=mask_path_1,
                output_data_dir_1=output_data_dir_1,
                output_mask_dir_1=output_mask_dir_1,
                min_mask_size_1=min_mask_size_1,
                max_mask_size_1=max_mask_size_1
            )
        else:
            print(f"跳过 {file_1}：掩码文件不存在")


def create_circular_kernel_2(radius_2):
    diameter_2 = 2 * radius_2 + 1
    kernel_2 = np.zeros((diameter_2, diameter_2), dtype=np.uint8)
    center_2 = (radius_2, radius_2)
    cv2.circle(kernel_2, center_2, radius_2, 1, -1)
    return kernel_2.astype(bool)


def dilate_segmentation_2d_2(segmentation_array_2):
    # 基础区域改为label=5，用于生成异常掩码的范围
    seg_mask_2 = (segmentation_array_2 == 5).astype(bool)
    struct_elem_2 = create_circular_kernel_2(radius_2=5)

    dilated_2 = binary_dilation(seg_mask_2, struct_elem_2, iterations=0).astype(np.uint8)
    # 限制在label=5区域内生成异常掩码的候选范围
    dilated_2[segmentation_array_2 != 5] = 0
    dilated_2 = binary_dilation(dilated_2.astype(bool), struct_elem_2, iterations=1).astype(np.uint8)
    # 确保异常掩码的候选范围在label=5内
    dilated_2 = dilated_2 * (segmentation_array_2 == 5).astype(np.uint8)

    return dilated_2


def generate_anomaly_mask_2d_2(dilate_mask_2, original_mask_2, height_2, width_2, min_size_2, max_size_2):
    anomaly_mask_2 = np.zeros((height_2, width_2), dtype=np.uint8)
    loop_num_2 = 0
    max_loops_2 = 50
    threshold_2 = 200

    while True:
        noise_2 = np.random.randint(0, 256, (height_2, width_2), dtype=np.uint8)
        blur_2 = gaussian_filter(noise_2.astype(np.float32), sigma=15)
        stretch_2 = skimage.exposure.rescale_intensity(blur_2, out_range=(0, 255)).astype(np.uint8)
        thresh_2 = (stretch_2 >= threshold_2).astype(np.uint8) * 255

        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        opened_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, kernel_2)
        closed_2 = cv2.morphologyEx(opened_2, cv2.MORPH_CLOSE, kernel_2)

        # 严格限制异常掩码在label=5区域内
        anomaly_mask_2 = (dilate_mask_2 * closed_2 * (original_mask_2 == 5)).astype(np.uint8)
        anomaly_mask_2 = np.where(anomaly_mask_2 > 0, 1, 0).astype(np.uint8)
        mask_size_2 = np.sum(anomaly_mask_2)

        if min_size_2 <= mask_size_2 <= max_size_2:
            break
        loop_num_2 += 1
        if loop_num_2 >= max_loops_2:
            print(f"警告：达到最大循环次数({max_loops_2})，无法生成合适掩码")
            break

    return anomaly_mask_2


def generate_stripes_mask_2(shape_2, anomaly_mask_2, density_2=0.12, width_range_2=(2, 5)):
    """生成异常区域内的无规则条纹掩码"""
    height_2, width_2 = shape_2
    stripes_2 = np.zeros(shape_2, dtype=np.uint8)

    # 随机生成条纹数量（基于异常区域大小）
    mask_size_2 = np.sum(anomaly_mask_2)
    num_stripes_2 = max(2, int(mask_size_2 * density_2 / 100))  # 降低密度，避免条纹过密

    for _ in range(num_stripes_2):
        # 随机选择条纹起点（必须在异常区域内）
        y_coords_2, x_coords_2 = np.where(anomaly_mask_2 == 1)
        if len(y_coords_2) == 0:
            continue  # 异常区域为空则跳过
        idx_2 = np.random.randint(0, len(y_coords_2))
        y1_2, x1_2 = y_coords_2[idx_2], x_coords_2[idx_2]

        # 随机生成条纹方向和长度
        angle_2 = np.random.uniform(0, np.pi)  # 0-180度随机角度
        length_2 = np.random.uniform(8, min(height_2, width_2) * 0.35)  # 条纹长度

        # 计算条纹终点
        y2_2 = int(y1_2 + length_2 * np.sin(angle_2))
        x2_2 = int(x1_2 + length_2 * np.cos(angle_2))

        # 确保终点在图像范围内
        y2_2 = np.clip(y2_2, 0, height_2 - 1)
        x2_2 = np.clip(x2_2, 0, width_2 - 1)

        # 随机条纹宽度
        stripe_width_2 = np.random.randint(*width_range_2)

        # 绘制条纹（使用抗锯齿线条）
        cv2.line(stripes_2, (x1_2, y1_2), (x2_2, y2_2), 1, stripe_width_2)

    # 确保条纹只在异常区域内（间接限制在label=5）
    stripes_2 = stripes_2 * anomaly_mask_2
    return stripes_2


def add_anomaly_to_image_2(image_2, anomaly_mask_2, original_mask_2):

    """
        仅将anomaly_mask和条纹限制在label=1处，其他膨胀区域不限制。
        外围出血透明度(异常颜色占比)设置为0.8，原图透明度占比0.2。
        """

    # --- 核心修改参数：外围透明度 ---
    # 异常颜色占比 (透明度)
    ANOMALY_ALPHA_2 = 0.8
    # 原图颜色占比
    ORIGINAL_ALPHA_2 = 1.0 - ANOMALY_ALPHA_2
    # -------------------------------

    """仅将anomaly_mask和条纹限制在label=5处，其他膨胀区域不限制"""
    # 严格限制异常掩码在label=5区域
    anomaly_mask_2 = anomaly_mask_2 * (original_mask_2 == 5).astype(np.uint8)

    # 生成条纹掩码（因基于anomaly_mask，故自动限制在label=5）
    stripes_mask_2 = generate_stripes_mask_2(shape_2=image_2.shape, anomaly_mask_2=anomaly_mask_2)

    # 条纹边缘模糊区域（因基于条纹掩码，故自动限制在label=5）
    stripe_blur_kernel_2 = create_circular_kernel_2(radius_2=3)
    stripes_mask_blur_area_2 = binary_dilation(stripes_mask_2.astype(bool), stripe_blur_kernel_2, iterations=3).astype(
        np.uint8)
    stripes_edge_area_2 = stripes_mask_blur_area_2 - stripes_mask_2

    # 核心修改：仅计算label>0区域的像素均值
    # 生成label>0的掩码
    valid_mask_2 = (original_mask_2 > 0).astype(bool)
    # 提取有效区域的像素
    valid_pixels_2 = image_2[valid_mask_2]
    rand_2 = np.random.uniform(0.6, 0.85)
    # 计算有效区域的均值，若无有效像素则使用默认值50
    data_mean_2 = np.mean(valid_pixels_2) if len(valid_pixels_2) > 0 else 50
    base_offset_2 = data_mean_2 * 1.15 * rand_2

    # 随机噪声缩放比例（用于基础异常区域）
    noise_choice_2 = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
    if noise_choice_2 == 1:
        base_scale_2 = np.random.uniform(0.1, 0.9, anomaly_mask_2.shape)
    elif noise_choice_2 == 2:
        base_scale_2 = np.random.uniform(0.2, 0.8, anomaly_mask_2.shape)
    else:
        base_scale_2 = np.random.uniform(0.3, 0.7, anomaly_mask_2.shape)

    # 计算基础异常区域颜色（仅在label=5内）
    base_anomaly_color_2 = anomaly_mask_2 * (image_2 * base_scale_2 + base_offset_2)

    # 基础异常区域叠加（仅在label=5内修改）
    hemo_image_2 = image_2.copy()
    hemo_image_2[original_mask_2 == 5] = (image_2[original_mask_2 == 5] * (1 - anomaly_mask_2[original_mask_2 == 5]) +
                                          base_anomaly_color_2[original_mask_2 == 5])

    # 第一次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem1_2 = create_circular_kernel_2(radius_2=1)
    dilate1_2 = binary_dilation(anomaly_mask_2.astype(bool), struct_elem1_2, iterations=1).astype(np.uint8)
    dilate1_2 = dilate1_2 * (original_mask_2 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate1_only_2 = dilate1_2 - anomaly_mask_2

    deep_scale1_2 = dilate1_only_2 * np.random.uniform(0.05, 0.3, dilate1_only_2.shape)
    deep_offset1_2 = dilate1_only_2 * (data_mean_2 * 1.1 * rand_2)
    # hemo_image_2 = hemo_image_2 * (1 - dilate1_only_2) + (image_2 * deep_scale1_2 + deep_offset1_2)
    anomaly_color1_2 = (image_2 * deep_scale1_2 + deep_offset1_2)

    # --- 第一次膨胀区域：应用 0.8 / 0.2 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1) + (image_1 * deep_scale1_1 + deep_offset1_1)
    # 修改为：新图像 = 旧图像 * (1 - 0.8*Mask) + 异常颜色 * 0.2*Mask
    hemo_image_2 = hemo_image_2 * (1 - dilate1_only_2 * ANOMALY_ALPHA_2) + anomaly_color1_2 * ANOMALY_ALPHA_2

    # 第二次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem2_2 = create_circular_kernel_2(radius_2=1)
    dilate2_2 = binary_dilation(dilate1_2.astype(bool), struct_elem2_2, iterations=2).astype(np.uint8)
    dilate2_2 = dilate2_2 * (original_mask_2 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate2_only_2 = dilate2_2 - dilate1_2

    deep_scale2_2 = np.random.uniform(0.02, 0.2, image_2.shape)
    deep_offset2_2 = data_mean_2 * 0.9
    # hemo_image_2 = hemo_image_2 * (1 - dilate2_only_2) + (
    #         image_2 * (dilate2_only_2 * deep_scale2_2) + (dilate2_only_2 * deep_offset2_2))
    anomaly_color2_2 = (image_2 * (dilate2_only_2 * deep_scale2_2) + (dilate2_only_2 * deep_offset2_2))

    # --- 第二次膨胀区域：应用 0.6 / 0.4 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1) + (image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))
    # 修改为：新图像 = 旧图像 * (1 - 0.6*Mask) + 异常颜色 * 0.6*Mask
    hemo_image_2 = hemo_image_2 * (1 - dilate2_only_2 * ANOMALY_ALPHA_2) + anomaly_color2_2 * ANOMALY_ALPHA_2

    # 关键修改4：将掩码为0的区域设为0（黑色背景）
    hemo_image_2[original_mask_2 == 0] = 0

    return np.clip(hemo_image_2, 0, 255).astype(np.uint8)


def process_single_image_2(data_path_2, mask_path_2, output_data_dir_2, output_mask_dir_2, min_mask_size_2,
                           max_mask_size_2):
    os.makedirs(output_data_dir_2, exist_ok=True)
    os.makedirs(output_mask_dir_2, exist_ok=True)

    try:
        with Image.open(data_path_2) as img_2:
            image_2 = np.array(img_2.convert('L'), dtype=np.float32)
        with Image.open(mask_path_2) as msk_2:
            original_mask_2 = np.array(msk_2.convert('L'), dtype=np.uint8)

        # 检查是否存在label=5的区域（因异常掩码需在此生成）
        if not np.any(original_mask_2 == 5):
            print(f"跳过 {os.path.basename(data_path_2)}：无值为5的掩码区域")
            return

        height_2, width_2 = image_2.shape
        print(f"处理 {os.path.basename(data_path_2)} ({height_2}x{width_2})")

        hemo_range_2 = dilate_segmentation_2d_2(original_mask_2)
        if np.sum(hemo_range_2) < min_mask_size_2:
            print(f"跳过 {os.path.basename(data_path_2)}：label=5区域内可生成异常的范围过小")
            return

        anomaly_mask_2 = generate_anomaly_mask_2d_2(
            hemo_range_2,
            original_mask_2,
            height_2,
            width_2,
            min_size_2=min_mask_size_2,
            max_size_2=max_mask_size_2
        )
        mask_size_2 = np.sum(anomaly_mask_2)
        if not (min_mask_size_2 <= mask_size_2 <= max_mask_size_2):
            print(f"跳过 {os.path.basename(data_path_2)}：掩码大小不符 ({mask_size_2}像素)")
            return

        hemo_image_2 = add_anomaly_to_image_2(image_2, anomaly_mask_2, original_mask_2)

        # 确保保存前掩码为0的区域已被清除
        hemo_image_2 = hemo_image_2 * (original_mask_2 > 0).astype(np.uint8)

        # 核心修改：对最终要保存的图像进行有效像素占比判断
        valid_flag, valid_ratio = check_valid_pixel_ratio(hemo_image_2)
        if not valid_flag:
            print(f"跳过 {os.path.basename(data_path_2)}：最终保存图像的有效像素占比{valid_ratio:.2%} < 30%")
            return

        # 核心修改：在文件名前添加"2_"前缀
        base_name_2 = os.path.basename(data_path_2)
        new_base_name_2 = f"2_{base_name_2}"  # 拼接生成带前缀的新文件名

        # 保存处理后的图片（使用带2_前缀的文件名）
        Image.fromarray(hemo_image_2).save(os.path.join(output_data_dir_2, new_base_name_2))
        # 保存掩码图片（使用带2_前缀的文件名）
        Image.fromarray((anomaly_mask_2 * 255).astype(np.uint8)).save(os.path.join(output_mask_dir_2, new_base_name_2))
        print(f"已保存 {new_base_name_2}（掩码大小: {mask_size_2}像素）")  # 打印新文件名

    except Exception as e_2:
        print(f"处理 {data_path_2} 失败: {e_2}")


def process_all_images_2(data_dir_2, mask_dir_2, output_data_dir_2, output_mask_dir_2, min_mask_size_2,
                         max_mask_size_2):
    for file_2 in [f_2 for f_2 in os.listdir(data_dir_2) if f_2.lower().endswith('.png')]:
        data_path_2 = os.path.join(data_dir_2, file_2)
        mask_path_2 = os.path.join(mask_dir_2, file_2)
        if os.path.exists(mask_path_2):
            process_single_image_2(
                data_path_2,
                mask_path_2,
                output_data_dir_2,
                output_mask_dir_2,
                min_mask_size_2,
                max_mask_size_2
            )
        else:
            print(f"跳过 {file_2}：掩码文件不存在")


def keep_largest_connected_component_2d_3(image_2d_3):
    if not np.any(image_2d_3):
        return image_2d_3

    labeled_3, num_labels_3 = label(image_2d_3)
    if num_labels_3 == 0:
        return image_2d_3

    component_sizes_3 = np.bincount(labeled_3.flatten())
    component_sizes_3[0] = 0
    max_label_3 = np.argmax(component_sizes_3)

    largest_component_3 = np.zeros_like(image_2d_3, dtype=image_2d_3.dtype)
    largest_component_3[labeled_3 == max_label_3] = image_2d_3[labeled_3 == max_label_3]
    return largest_component_3


def create_circular_kernel_3(radius_3):
    diameter_3 = 2 * radius_3 + 1
    kernel_3 = np.zeros((diameter_3, diameter_3), dtype=np.uint8)
    center_3 = (radius_3, radius_3)
    cv2.circle(kernel_3, center_3, radius_3, 1, -1)
    return kernel_3.astype(bool)


def dilate_segmentation_2d_3(segmentation_array_3):
    # 基础区域为label=2，用于生成异常掩码的范围
    seg_mask_3 = (segmentation_array_3 == 2).astype(bool)
    struct_elem_3 = create_circular_kernel_3(radius_3=15)

    dilated_3 = binary_dilation(seg_mask_3, struct_elem_3, iterations=1).astype(np.uint8)
    # 限制在label=2区域内生成异常掩码的候选范围
    dilated_3[segmentation_array_3 != 2] = 0
    dilated_3 = binary_dilation(dilated_3.astype(bool), struct_elem_3, iterations=1).astype(np.uint8)
    # 确保异常掩码的候选范围在label=2内
    dilated_3 = dilated_3 * (segmentation_array_3 == 2).astype(np.uint8)

    return keep_largest_connected_component_2d_3(dilated_3)


def generate_anomaly_mask_2d_3(dilate_mask_3, original_mask_3, height_3, width_3, min_size_3, max_size_3):
    anomaly_mask_3 = np.zeros((height_3, width_3), dtype=np.uint8)
    loop_num_3 = 0
    max_loops_3 = 50
    threshold_3 = 200

    while True:
        noise_3 = np.random.randint(0, 256, (height_3, width_3), dtype=np.uint8)
        blur_3 = gaussian_filter(noise_3.astype(np.float32), sigma=15)
        stretch_3 = skimage.exposure.rescale_intensity(blur_3, out_range=(0, 255)).astype(np.uint8)
        thresh_3 = (stretch_3 >= threshold_3).astype(np.uint8) * 255

        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_OPEN, kernel_3)
        closed_3 = cv2.morphologyEx(opened_3, cv2.MORPH_CLOSE, kernel_3)

        # 严格限制异常掩码在label=2区域内
        anomaly_mask_3 = (dilate_mask_3 * closed_3 * (original_mask_3 == 2)).astype(np.uint8)
        anomaly_mask_3 = np.where(anomaly_mask_3 > 0, 1, 0).astype(np.uint8)
        mask_size_3 = np.sum(anomaly_mask_3)

        if min_size_3 <= mask_size_3 <= max_size_3:
            break
        loop_num_3 += 1
        if loop_num_3 >= max_loops_3:
            print(f"警告：达到最大循环次数({max_loops_3})，无法生成合适掩码")
            break

    return anomaly_mask_3


def generate_stripes_mask_3(shape_3, anomaly_mask_3, density_3=0.12, width_range_3=(2, 5)):
    """生成异常区域内的无规则条纹掩码"""
    height_3, width_3 = shape_3
    stripes_3 = np.zeros(shape_3, dtype=np.uint8)

    # 随机生成条纹数量（基于异常区域大小）
    mask_size_3 = np.sum(anomaly_mask_3)
    num_stripes_3 = max(2, int(mask_size_3 * density_3 / 100))  # 降低密度，避免条纹过密

    for _ in range(num_stripes_3):
        # 随机选择条纹起点（必须在异常区域内）
        y_coords_3, x_coords_3 = np.where(anomaly_mask_3 == 1)
        if len(y_coords_3) == 0:
            continue  # 异常区域为空则跳过
        idx_3 = np.random.randint(0, len(y_coords_3))
        y1_3, x1_3 = y_coords_3[idx_3], x_coords_3[idx_3]

        # 随机生成条纹方向和长度
        angle_3 = np.random.uniform(0, np.pi)  # 0-180度随机角度
        length_3 = np.random.uniform(8, min(height_3, width_3) * 0.35)  # 条纹长度

        # 计算条纹终点
        y2_3 = int(y1_3 + length_3 * np.sin(angle_3))
        x2_3 = int(x1_3 + length_3 * np.cos(angle_3))

        # 确保终点在图像范围内
        y2_3 = np.clip(y2_3, 0, height_3 - 1)
        x2_3 = np.clip(x2_3, 0, width_3 - 1)

        # 随机条纹宽度
        stripe_width_3 = np.random.randint(*width_range_3)

        # 绘制条纹（使用抗锯齿线条）
        cv2.line(stripes_3, (x1_3, y1_3), (x2_3, y2_3), 1, stripe_width_3)

    # 确保条纹只在异常区域内（间接限制在label=2）
    stripes_3 = stripes_3 * anomaly_mask_3
    return stripes_3


def add_anomaly_to_image_3(image_3, anomaly_mask_3, original_mask_3):

    """
        仅将anomaly_mask和条纹限制在label=2处，其他膨胀区域不限制。
        外围出血透明度(异常颜色占比)设置为0.8，原图透明度占比0.2。
        """

    # --- 核心修改参数：外围透明度 ---
    # 异常颜色占比 (透明度)
    ANOMALY_ALPHA_3 = 0.8
    # 原图颜色占比
    ORIGINAL_ALPHA_3 = 1.0 - ANOMALY_ALPHA_3
    # -------------------------------

    """仅将anomaly_mask和条纹限制在label=2处，其他膨胀区域不限制"""
    # 核心：严格限制异常掩码在label=2区域
    anomaly_mask_3 = anomaly_mask_3 * (original_mask_3 == 2).astype(np.uint8)

    # 生成条纹掩码（因基于anomaly_mask，故自动限制在label=2）
    stripes_mask_3 = generate_stripes_mask_3(shape_3=image_3.shape, anomaly_mask_3=anomaly_mask_3)

    # 条纹边缘模糊区域（因基于条纹掩码，故自动限制在label=2）
    stripe_blur_kernel_3 = create_circular_kernel_3(radius_3=1)
    stripes_mask_blur_area_3 = binary_dilation(stripes_mask_3.astype(bool), stripe_blur_kernel_3, iterations=3).astype(
        np.uint8)
    stripes_edge_area_3 = stripes_mask_blur_area_3 - stripes_mask_3

    # 核心修改：仅计算label>0区域的像素均值
    # 生成label>0的掩码
    valid_mask_3 = (original_mask_3 > 0).astype(bool)
    # 提取有效区域的像素
    valid_pixels_3 = image_3[valid_mask_3]
    rand_3 = np.random.uniform(1.0, 1.2)
    # 计算有效区域的均值，若无有效像素则使用默认值50
    data_mean_3 = np.mean(valid_pixels_3) if len(valid_pixels_3) > 0 else 50
    base_offset_3 = data_mean_3 * 1.2 * rand_3

    # 随机噪声缩放比例（用于基础异常区域）
    noise_choice_3 = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
    if noise_choice_3 == 1:
        base_scale_3 = np.random.uniform(0.1, 0.9, anomaly_mask_3.shape)
    elif noise_choice_3 == 2:
        base_scale_3 = np.random.uniform(0.2, 0.8, anomaly_mask_3.shape)
    else:
        base_scale_3 = np.random.uniform(0.3, 0.7, anomaly_mask_3.shape)

    # 计算基础异常区域颜色（仅在label=2内）
    base_anomaly_color_3 = anomaly_mask_3 * (image_3 * base_scale_3 + base_offset_3)

    # 基础异常区域叠加（仅在label=2内修改）
    hemo_image_3 = image_3.copy()
    hemo_image_3[original_mask_3 == 2] = (image_3[original_mask_3 == 2] * (1 - anomaly_mask_3[original_mask_3 == 2]) +
                                          base_anomaly_color_3[original_mask_3 == 2])

    # 第一次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem1_3 = create_circular_kernel_3(radius_3=1)
    dilate1_3 = binary_dilation(anomaly_mask_3.astype(bool), struct_elem1_3, iterations=1).astype(np.uint8)
    dilate1_3 = dilate1_3 * (original_mask_3 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate1_only_3 = dilate1_3 - anomaly_mask_3

    deep_scale1_3 = dilate1_only_3 * np.random.uniform(0.05, 0.3, dilate1_only_3.shape)
    deep_offset1_3 = dilate1_only_3 * (data_mean_3 * 1.1 * rand_3)
    # hemo_image_3 = hemo_image_3 * (1 - dilate1_only_3) + (image_3 * deep_scale1_3 + deep_offset1_3)
    anomaly_color1_3 = (image_3 * deep_scale1_3 + deep_offset1_3)

    # --- 第一次膨胀区域：应用 0.6 / 0.4 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1) + (image_1 * deep_scale1_1 + deep_offset1_1)
    # 修改为：新图像 = 旧图像 * (1 - 0.6*Mask) + 异常颜色 * 0.6*Mask
    hemo_image_3 = hemo_image_3 * (1 - dilate1_only_3 * ANOMALY_ALPHA_3) + anomaly_color1_3 * ANOMALY_ALPHA_3

    # 第二次膨胀区域处理（不限制label，仅限制在原始掩码有效区域）
    struct_elem2_3 = create_circular_kernel_3(radius_3=1)
    dilate2_3 = binary_dilation(dilate1_3.astype(bool), struct_elem2_3, iterations=2).astype(np.uint8)
    dilate2_3 = dilate2_3 * (original_mask_3 > 0).astype(np.uint8)  # 仅限制在有效掩码区域
    dilate2_only_3 = dilate2_3 - dilate1_3

    deep_scale2_3 = np.random.uniform(0.02, 0.2, image_3.shape)
    deep_offset2_3 = data_mean_3 * 0.9
    # hemo_image_3 = hemo_image_3 * (1 - dilate2_only_3) + (
    #         image_3 * (dilate2_only_3 * deep_scale2_3) + (dilate2_only_3 * deep_offset2_3))
    anomaly_color2_3 = (image_3 * (dilate2_only_3 * deep_scale2_3) + (dilate2_only_3 * deep_offset2_3))

    # --- 第二次膨胀区域：应用 0.6 / 0.4 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1) + (image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))
    # 修改为：新图像 = 旧图像 * (1 - 0.6*Mask) + 异常颜色 * 0.6*Mask
    hemo_image_3 = hemo_image_3 * (1 - dilate2_only_3 * ANOMALY_ALPHA_3) + anomaly_color2_3 * ANOMALY_ALPHA_3

    # 条纹叠加（仅在label=2内）
    stripes_base_color_3 = base_anomaly_color_3 * stripes_mask_3
    stripes_color_3 = stripes_base_color_3 * 1.2
    hemo_image_3 = hemo_image_3 * (1 - stripes_mask_3) + stripes_color_3

    # 条纹边缘模糊处理（仅在label=2内）
    if np.sum(stripes_edge_area_3) > 0:
        edge_blur_3 = gaussian_filter(hemo_image_3, sigma=1.0)
        hemo_image_3 = hemo_image_3 * (1 - stripes_edge_area_3 * 1.2) + edge_blur_3 * (stripes_edge_area_3 * 1.2)

    # 整体模糊
    # hemo_image_3 = gaussian_filter(hemo_image_3, sigma=1.2)

    # 关键修改1：将掩码为0的区域设为0（黑色背景）
    hemo_image_3[original_mask_3 == 0] = 0

    return np.clip(hemo_image_3, 0, 255).astype(np.uint8)


def process_single_image_3(data_path_3, mask_path_3, output_data_dir_3, output_mask_dir_3, min_mask_size_3,
                           max_mask_size_3):
    os.makedirs(output_data_dir_3, exist_ok=True)
    os.makedirs(output_mask_dir_3, exist_ok=True)

    try:
        with Image.open(data_path_3) as img_3:
            image_3 = np.array(img_3.convert('L'), dtype=np.float32)
        with Image.open(mask_path_3) as msk_3:
            original_mask_3 = np.array(msk_3.convert('L'), dtype=np.uint8)

        # 检查是否存在label=2的区域（因异常掩码需在此生成）
        if not np.any(original_mask_3 == 2):
            print(f"跳过 {os.path.basename(data_path_3)}：无值为2的掩码区域")
            return

        height_3, width_3 = image_3.shape
        print(f"处理 {os.path.basename(data_path_3)} ({height_3}x{width_3})")

        hemo_range_3 = dilate_segmentation_2d_3(original_mask_3)
        if np.sum(hemo_range_3) < min_mask_size_3:
            print(f"跳过 {os.path.basename(data_path_3)}：label=2区域内可生成异常的范围过小")
            return

        anomaly_mask_3 = generate_anomaly_mask_2d_3(
            hemo_range_3,
            original_mask_3,
            height_3,
            width_3,
            min_size_3=min_mask_size_3,
            max_size_3=max_mask_size_3
        )
        mask_size_3 = np.sum(anomaly_mask_3)
        if not (min_mask_size_3 <= mask_size_3 <= max_mask_size_3):
            print(f"跳过 {os.path.basename(data_path_3)}：掩码大小不符 ({mask_size_3}像素)")
            return

        hemo_image_3 = add_anomaly_to_image_3(image_3, anomaly_mask_3, original_mask_3)

        # 关键修改2：确保保存前掩码为0的区域已被清除
        # （双重保障，与add_anomaly_to_image中的处理一致）
        hemo_image_3 = hemo_image_3 * (original_mask_3 > 0).astype(np.uint8)

        # 核心修改：对最终要保存的图像进行有效像素占比判断
        valid_flag, valid_ratio = check_valid_pixel_ratio(hemo_image_3)
        if not valid_flag:
            print(f"跳过 {os.path.basename(data_path_3)}：最终保存图像的有效像素占比{valid_ratio:.2%} < 30%")
            return

        # 核心修改：在文件名前添加"3_"前缀
        base_name_3 = os.path.basename(data_path_3)
        new_base_name_3 = f"3_{base_name_3}"  # 拼接生成带前缀的新文件名

        # 保存处理后的图片（使用带3_前缀的文件名）
        Image.fromarray(hemo_image_3).save(os.path.join(output_data_dir_3, new_base_name_3))
        # 保存掩码图片（使用带3_前缀的文件名）
        Image.fromarray((anomaly_mask_3 * 255).astype(np.uint8)).save(os.path.join(output_mask_dir_3, new_base_name_3))
        print(f"已保存 {new_base_name_3}（掩码大小: {mask_size_3}像素）")  # 打印新文件名

    except Exception as e_3:
        print(f"处理 {data_path_3} 失败: {e_3}")


def process_all_images_3(data_dir_3, mask_dir_3, output_data_dir_3, output_mask_dir_3, min_mask_size_3,
                         max_mask_size_3):
    for file_3 in [f_3 for f_3 in os.listdir(data_dir_3) if f_3.lower().endswith('.png')]:
        data_path_3 = os.path.join(data_dir_3, file_3)
        mask_path_3 = os.path.join(mask_dir_3, file_3)
        if os.path.exists(mask_path_3):
            process_single_image_3(
                data_path_3,
                mask_path_3,
                output_data_dir_3,
                output_mask_dir_3,
                min_mask_size_3,
                max_mask_size_3
            )
        else:
            print(f"跳过 {file_3}：掩码文件不存在")


def keep_largest_connected_component_2d_4(image_2d_4):
    if not np.any(image_2d_4):
        return image_2d_4

    labeled_4, num_labels_4 = label(image_2d_4)
    if num_labels_4 == 0:
        return image_2d_4

    component_sizes_4 = np.bincount(labeled_4.flatten())
    component_sizes_4[0] = 0
    max_label_4 = np.argmax(component_sizes_4)

    largest_component_4 = np.zeros_like(image_2d_4, dtype=image_2d_4.dtype)
    largest_component_4[labeled_4 == max_label_4] = image_2d_4[labeled_4 == max_label_4]
    return largest_component_4


def create_circular_kernel_4(radius_4):
    diameter_4 = 2 * radius_4 + 1
    kernel_4 = np.zeros((diameter_4, diameter_4), dtype=np.uint8)
    center_4 = (radius_4, radius_4)
    cv2.circle(kernel_4, center_4, radius_4, 1, -1)
    return kernel_4.astype(bool)


def dilate_segmentation_2d_4(segmentation_array_4):
    """生成label=4的膨胀区域作为异常候选范围（减小膨胀半径，限制候选范围大小）"""
    seg_mask_4 = (segmentation_array_4 == 4).astype(bool)
    struct_elem_4 = create_circular_kernel_4(radius_4=3)  # 关键：从10减小到3，大幅缩小膨胀候选范围

    # 第一次膨胀：基于label=4向周围扩展
    dilated_4 = binary_dilation(seg_mask_4, struct_elem_4, iterations=1).astype(np.uint8)
    # 第二次膨胀：进一步扩大候选范围（不再限制在label=4内）
    dilated_4 = binary_dilation(dilated_4.astype(bool), struct_elem_4, iterations=1).astype(np.uint8)
    # 仅限制在原始掩码的非0区域（避免超出有效区域）
    dilated_4 = dilated_4 * (segmentation_array_4 > 0).astype(np.uint8)

    return keep_largest_connected_component_2d_4(dilated_4)


def generate_anomaly_mask_2d_4(dilate_mask_4, original_mask_4, height_4, width_4, min_size_4, max_size_4):
    """生成与label=4有交集的异常掩码（缩小掩码尺寸，调整阈值让掩码更紧凑）"""
    anomaly_mask_4 = np.zeros((height_4, width_4), dtype=np.uint8)
    loop_num_4 = 0
    max_loops_4 = 100
    threshold_4 = 220  # 关键：从200提高到220，生成的掩码白色区域更少，尺寸更小
    label4_mask_4 = (original_mask_4 == 4).astype(np.uint8)  # label=4的掩码

    while True:
        noise_4 = np.random.randint(0, 256, (height_4, width_4), dtype=np.uint8)
        blur_4 = gaussian_filter(noise_4.astype(np.float32), sigma=18)  # 关键：从15增大到18，模糊更强，掩码更分散但整体更小
        stretch_4 = skimage.exposure.rescale_intensity(blur_4, out_range=(0, 255)).astype(np.uint8)
        thresh_4 = (stretch_4 >= threshold_4).astype(np.uint8) * 255

        # 关键：减小形态学核大小，从9x9改为5x5，让掩码更细碎、尺寸更小
        kernel_4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened_4 = cv2.morphologyEx(thresh_4, cv2.MORPH_OPEN, kernel_4)
        closed_4 = cv2.morphologyEx(opened_4, cv2.MORPH_CLOSE, kernel_4)

        # 异常掩码基于膨胀后的候选范围（允许扩展到label=4外）
        anomaly_mask_4 = (dilate_mask_4 * closed_4).astype(np.uint8)
        anomaly_mask_4 = np.where(anomaly_mask_4 > 0, 1, 0).astype(np.uint8)
        mask_size_4 = np.sum(anomaly_mask_4)

        # 核心条件：掩码大小符合要求 + 与label=4有交集
        has_intersection_4 = np.sum(anomaly_mask_4 * label4_mask_4) > 0
        if min_size_4 <= mask_size_4 <= max_size_4 and has_intersection_4:
            break
        loop_num_4 += 1
        if loop_num_4 >= max_loops_4:
            print(f"警告：达到最大循环次数({max_loops_4})，无法生成与label=4有交集的合适掩码")
            break

    return anomaly_mask_4


def generate_stripes_mask_4(shape_4, anomaly_mask_4, density_4=0.05, width_range_4=(1, 3)):
    """生成异常区域内的无规则条纹掩码（减小密度和宽度，减少条纹数量/粗细）"""
    height_4, width_4 = shape_4
    stripes_4 = np.zeros(shape_4, dtype=np.uint8)

    # 随机生成条纹数量（基于异常区域大小）
    mask_size_4 = np.sum(anomaly_mask_4)
    num_stripes_4 = max(1, int(mask_size_4 * density_4 / 100))  # 关键：max从2改1，减少最少条纹数；密度从0.12改0.05，大幅减少条纹数

    for _ in range(num_stripes_4):
        # 随机选择条纹起点（必须在异常区域内）
        y_coords_4, x_coords_4 = np.where(anomaly_mask_4 == 1)
        if len(y_coords_4) == 0:
            continue  # 异常区域为空则跳过
        idx_4 = np.random.randint(0, len(y_coords_4))
        y1_4, x1_4 = y_coords_4[idx_4], x_coords_4[idx_4]

        # 随机生成条纹方向和长度（减小长度范围，让条纹更短）
        angle_4 = np.random.uniform(0, np.pi)  # 0-180度随机角度
        length_4 = np.random.uniform(4, min(height_4, width_4) * 0.2)  # 关键：从8改4，0.35改0.2，缩短条纹长度

        # 计算条纹终点
        y2_4 = int(y1_4 + length_4 * np.sin(angle_4))
        x2_4 = int(x1_4 + length_4 * np.cos(angle_4))

        # 确保终点在图像范围内
        y2_4 = np.clip(y2_4, 0, height_4 - 1)
        x2_4 = np.clip(x2_4, 0, width_4 - 1)

        # 随机条纹宽度（减小宽度范围，从2-5改1-3）
        stripe_width_4 = np.random.randint(*width_range_4)

        # 绘制条纹（使用抗锯齿线条）
        cv2.line(stripes_4, (x1_4, y1_4), (x2_4, y2_4), 1, stripe_width_4)

    # 确保条纹只在异常区域内
    stripes_4 = stripes_4 * anomaly_mask_4
    return stripes_4


def add_anomaly_to_image_4(image_4, anomaly_mask_4, original_mask_4):

    """
        仅将anomaly_mask和条纹限制在label=4处，其他膨胀区域不限制。
        外围出血透明度(异常颜色占比)设置为0.8，原图透明度占比0.2。
        """

    # --- 核心修改参数：外围透明度 ---
    # 异常颜色占比 (透明度)
    ANOMALY_ALPHA_4 = 0.8
    # 原图颜色占比
    ORIGINAL_ALPHA_4 = 1.0 - ANOMALY_ALPHA_4
    # -------------------------------

    """添加异常区域（减小膨胀处理的范围，让异常边缘扩展更小）"""
    # 生成无规则条纹掩码（随异常掩码扩展）
    stripes_mask_4 = generate_stripes_mask_4(image_4.shape, anomaly_mask_4)

    # 条纹边缘模糊区域设置（减小膨胀半径，从3改1，减少模糊范围）
    stripe_blur_kernel_4 = create_circular_kernel_4(radius_4=1)
    stripes_mask_blur_area_4 = binary_dilation(stripes_mask_4.astype(bool), stripe_blur_kernel_4, iterations=2).astype(
        np.uint8)  # 迭代从3改2，减少模糊迭代
    stripes_edge_area_4 = stripes_mask_blur_area_4 - stripes_mask_4

    # 计算基础异常区域参数（用label=4区域的像素均值，保证异常颜色贴合label=4特征）
    label4_pixels_4 = image_4[original_mask_4 == 4]
    data_mean_4 = np.mean(label4_pixels_4) if len(label4_pixels_4) > 0 else 50
    base_offset_4 = data_mean_4 * 0.9

    # 随机噪声缩放比例（用于基础异常区域）
    noise_choice_4 = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
    if noise_choice_4 == 1:
        base_scale_4 = np.random.uniform(0.1, 0.9, anomaly_mask_4.shape)
    elif noise_choice_4 == 2:
        base_scale_4 = np.random.uniform(0.2, 0.8, anomaly_mask_4.shape)
    else:
        base_scale_4 = np.random.uniform(0.3, 0.7, anomaly_mask_4.shape)

    # 计算基础异常区域颜色
    base_anomaly_color_4 = anomaly_mask_4 * (image_4 * base_scale_4 + base_offset_4)

    # 基础异常区域叠加（直接覆盖异常掩码区域，允许扩展到label=4外）
    hemo_image_4 = image_4 * (1 - anomaly_mask_4) + base_anomaly_color_4

    # 第一次膨胀区域处理（减小膨胀半径，从2改1，限制膨胀范围）
    struct_elem1_4 = create_circular_kernel_4(radius_4=1)
    dilate1_4 = binary_dilation(anomaly_mask_4.astype(bool), struct_elem1_4, iterations=1).astype(np.uint8)
    dilate1_4 = dilate1_4 * (original_mask_4 > 0).astype(np.uint8)  # 仅限制在有效区域
    dilate1_only_4 = dilate1_4 - anomaly_mask_4

    deep_scale1_4 = dilate1_only_4 * np.random.uniform(0.05, 0.3, dilate1_only_4.shape)
    deep_offset1_4 = dilate1_only_4 * (data_mean_4 * 0.7)
    # hemo_image_4 = hemo_image_4 * (1 - dilate1_only_4) + (image_4 * deep_scale1_4 + deep_offset1_4)
    anomaly_color1_4 = (image_4 * deep_scale1_4 + deep_offset1_4)

    # --- 第一次膨胀区域：应用 0.8 / 0.2 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate1_only_1) + (image_1 * deep_scale1_1 + deep_offset1_1)
    # 修改为：新图像 = 旧图像 * (1 - 0.8*Mask) + 异常颜色 * 0.8*Mask
    hemo_image_4 = hemo_image_4 * (1 - dilate1_only_4 * ANOMALY_ALPHA_4) + anomaly_color1_4 * ANOMALY_ALPHA_4

    # 第二次膨胀区域处理（减小迭代次数，从2改1，减少膨胀范围）
    struct_elem2_4 = create_circular_kernel_4(radius_4=1)
    dilate2_4 = binary_dilation(dilate1_4.astype(bool), struct_elem2_4, iterations=1).astype(np.uint8)
    dilate2_4 = dilate2_4 * (original_mask_4 > 0).astype(np.uint8)  # 仅限制在有效区域
    dilate2_only_4 = dilate2_4 - dilate1_4

    deep_scale2_4 = np.random.uniform(0.02, 0.2, image_4.shape)
    deep_offset2_4 = data_mean_4 * 0.8
    # hemo_image_4 = hemo_image_4 * (1 - dilate2_only_4) + (
    #         image_4 * (dilate2_only_4 * deep_scale2_4) + (dilate2_only_4 * deep_offset2_4))

    anomaly_color2_4 = (image_4 * (dilate2_only_4 * deep_scale2_4) + (dilate2_only_4 * deep_offset2_4))

    # --- 第二次膨胀区域：应用 0.6 / 0.4 透明度 ---
    # hemo_image_1 = hemo_image_1 * (1 - dilate2_only_1) + (image_1 * (dilate2_only_1 * deep_scale2_1) + (dilate2_only_1 * deep_offset2_1))
    # 修改为：新图像 = 旧图像 * (1 - 0.6*Mask) + 异常颜色 * 0.6*Mask
    hemo_image_4 = hemo_image_4 * (1 - dilate2_only_4 * ANOMALY_ALPHA_4) + anomaly_color2_4 * ANOMALY_ALPHA_4

    # 条纹颜色设置为基础掩码区域颜色的80%
    stripes_base_color_4 = base_anomaly_color_4 * stripes_mask_4
    stripes_color_4 = stripes_base_color_4 * 0.8
    hemo_image_4 = hemo_image_4 * (1 - stripes_mask_4) + stripes_color_4

    # 条纹边缘模糊处理
    if np.sum(stripes_edge_area_4) > 0:
        edge_blur_4 = gaussian_filter(hemo_image_4, sigma=1.0)
        hemo_image_4 = hemo_image_4 * (1 - stripes_edge_area_4 * 0.8) + edge_blur_4 * (stripes_edge_area_4 * 0.8)

    # 强制将掩码为0的区域设置为0（黑色背景）
    hemo_image_4[original_mask_4 == 0] = 0

    return np.clip(hemo_image_4, 0, 255).astype(np.uint8)


def process_single_image_4(data_path_4, mask_path_4, output_data_dir_4, output_mask_dir_4, min_mask_size_4,
                           max_mask_size_4):
    os.makedirs(output_data_dir_4, exist_ok=True)
    os.makedirs(output_mask_dir_4, exist_ok=True)

    try:
        with Image.open(data_path_4) as img_4:
            image_4 = np.array(img_4.convert('L'), dtype=np.float32)
        with Image.open(mask_path_4) as msk_4:
            original_mask_4 = np.array(msk_4.convert('L'), dtype=np.uint8)

        # 检查是否存在label=4的区域
        if not np.any(original_mask_4 == 4):
            print(f"跳过 {os.path.basename(data_path_4)}：无值为4的掩码区域")
            return

        height_4, width_4 = image_4.shape
        print(f"处理 {os.path.basename(data_path_4)} ({height_4}x{width_4})")

        hemo_range_4 = dilate_segmentation_2d_4(original_mask_4)
        if np.sum(hemo_range_4) < min_mask_size_4:
            print(f"跳过 {os.path.basename(data_path_4)}：label=4的膨胀候选范围过小")
            return

        anomaly_mask_4 = generate_anomaly_mask_2d_4(
            hemo_range_4,
            original_mask_4,
            height_4,
            width_4,
            min_size_4=min_mask_size_4,
            max_size_4=max_mask_size_4
        )
        mask_size_4 = np.sum(anomaly_mask_4)
        # 验证掩码是否与label=4有交集
        has_intersection_4 = np.sum(anomaly_mask_4 * (original_mask_4 == 4).astype(np.uint8)) > 0
        if not (min_mask_size_4 <= mask_size_4 <= max_mask_size_4) or not has_intersection_4:
            print(f"跳过 {os.path.basename(data_path_4)}：掩码大小不符或与label=4无交集 ({mask_size_4}像素)")
            return

        hemo_image_4 = add_anomaly_to_image_4(image_4, anomaly_mask_4, original_mask_4)

        # 双重保障：确保掩码为0的区域被清除
        hemo_image_4 = hemo_image_4 * (original_mask_4 > 0).astype(np.uint8)

        # 核心修改：在文件名前添加"4_"前缀
        base_name_4 = os.path.basename(data_path_4)
        new_base_name_4 = f"4_{base_name_4}"  # 拼接生成带前缀的新文件名

        # 核心修改：对最终要保存的图像进行有效像素占比判断
        valid_flag, valid_ratio = check_valid_pixel_ratio(hemo_image_4)
        if not valid_flag:
            print(f"跳过 {os.path.basename(data_path_4)}：最终保存图像的有效像素占比{valid_ratio:.2%} < 30%")
            return

        # 保存处理后的图片（使用带4_前缀的文件名）
        Image.fromarray(hemo_image_4).save(os.path.join(output_data_dir_4, new_base_name_4))
        # 保存掩码图片（使用带4_前缀的文件名）
        Image.fromarray((anomaly_mask_4 * 255).astype(np.uint8)).save(os.path.join(output_mask_dir_4, new_base_name_4))
        # 打印日志同步使用新文件名
        print(
            f"已保存 {new_base_name_4}（掩码大小: {mask_size_4}像素，与label=4交集像素数: {np.sum(anomaly_mask_4 * (original_mask_4 == 4).astype(np.uint8))}）")

    except Exception as e_4:
        print(f"处理 {data_path_4} 失败: {e_4}")


def process_all_images_4(data_dir_4, mask_dir_4, output_data_dir_4, output_mask_dir_4, min_mask_size_4,
                         max_mask_size_4):
    for file_4 in [f_4 for f_4 in os.listdir(data_dir_4) if f_4.lower().endswith('.png')]:
        data_path_4 = os.path.join(data_dir_4, file_4)
        mask_path_4 = os.path.join(mask_dir_4, file_4)
        if os.path.exists(mask_path_4):
            process_single_image_4(
                data_path_4,
                mask_path_4,
                output_data_dir_4,
                output_mask_dir_4,
                min_mask_size_4,
                max_mask_size_4
            )
        else:
            print(f"跳过 {file_4}：掩码文件不存在")


import random


def random_0_to_4_weighted():
    """按指定概率生成0-4的整数"""
    nums = [0, 1, 2, 3, 4]
    # 对应nums的概率权重（总和为1，也可使用[3,2.5,2,1.5,1]等比例值）
    weights = [0.08, 0.23, 0.23, 0.23, 0.23]
    # choices返回列表，取第一个元素
    return random.choices(nums, weights=weights)[0]


def process_all_images(data_dir, mask_dir, new_mask_dir, output_data_dir, output_mask_dir):
    MIN_MASK_SIZE_0 = 10
    MAX_MASK_SIZE_0 = 300

    MIN_MASK_SIZE_1 = 10
    MAX_MASK_SIZE_1 = 300

    MIN_MASK_SIZE_2 = 40
    MAX_MASK_SIZE_2 = 5000

    MIN_MASK_SIZE_3 = 10
    MAX_MASK_SIZE_3 = 300

    MIN_MASK_SIZE_4 = 40
    MAX_MASK_SIZE_4 = 5000

    for file in [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]:
        data_path = os.path.join(data_dir, file)
        mask_path = os.path.join(mask_dir, file)
        new_mask_path = os.path.join(new_mask_dir, file)

        num = random_0_to_4_weighted()

        if num == 0:
            if os.path.exists(mask_path):
                process_single_image_0(
                    data_path_0=data_path,
                    mask_path_0=mask_path,
                    output_data_dir_0=output_data_dir,
                    output_mask_dir_0=output_mask_dir,
                    min_mask_size_0=MIN_MASK_SIZE_0,
                    max_mask_size_0=MAX_MASK_SIZE_0
                )
            else:
                print(f"跳过 {file}：掩码文件不存在")

        elif num == 1:
            if os.path.exists(mask_path):
                process_single_image_1(
                    data_path_1=data_path,
                    mask_path_1=mask_path,
                    output_data_dir_1=output_data_dir,
                    output_mask_dir_1=output_mask_dir,
                    min_mask_size_1=MIN_MASK_SIZE_1,
                    max_mask_size_1=MAX_MASK_SIZE_1
                )
            else:
                print(f"跳过 {file}：掩码文件不存在")

        elif num == 2:
            if os.path.exists(mask_path):
                process_single_image_2(
                    data_path_2=data_path,
                    mask_path_2=new_mask_path,
                    output_data_dir_2=output_data_dir,
                    output_mask_dir_2=output_mask_dir,
                    min_mask_size_2=MIN_MASK_SIZE_2,
                    max_mask_size_2=MAX_MASK_SIZE_2
                )
            else:
                print(f"跳过 {file}：掩码文件不存在")

        elif num == 3:
            if os.path.exists(mask_path):
                process_single_image_3(
                    data_path_3=data_path,
                    mask_path_3=mask_path,
                    output_data_dir_3=output_data_dir,
                    output_mask_dir_3=output_mask_dir,
                    min_mask_size_3=MIN_MASK_SIZE_3,
                    max_mask_size_3=MAX_MASK_SIZE_3
                )
            else:
                print(f"跳过 {file}：掩码文件不存在")

        elif num == 4:
            if os.path.exists(mask_path):
                process_single_image_4(
                    data_path_4=data_path,
                    mask_path_4=mask_path,
                    output_data_dir_4=output_data_dir,
                    output_mask_dir_4=output_mask_dir,
                    min_mask_size_4=MIN_MASK_SIZE_4,
                    max_mask_size_4=MAX_MASK_SIZE_4
                )
            else:
                print(f"跳过 {file}：掩码文件不存在")


if __name__ == "__main__":
    DATA_DIR = "save/image2"
    MASK_DIR = "save/mask2"
    new_MASK_DIR = "save/new_mask0223"
    OUTPUT_DATA_DIR = "save/final0223"
    OUTPUT_MASK_DIR = "save/final_mask0223"

    process_all_images(DATA_DIR, MASK_DIR, new_MASK_DIR, OUTPUT_DATA_DIR, OUTPUT_MASK_DIR)
    print("所有图像处理完成")
