import os
import cv2
import numpy as np
from glob import glob

# 配置路径
DATA_DIR = "save/data2"
MASK_DIR = "save/mask2"
NEW_MASK_DIR = "save/new_mask0223"
MASK_VIEW_DIR = "save/mask_view0223"
# 创建目录（若不存在）
os.makedirs(NEW_MASK_DIR, exist_ok=True)
os.makedirs(MASK_VIEW_DIR, exist_ok=True)

# 定义临时标签和目标标签
TEMP_LABEL = 10  # 第一版条纹的临时标签
TARGET_LABEL = 5  # 最终目标标签


def load_mask_files(mask_dir):
    """加载掩码目录下所有PNG文件的路径"""
    mask_paths = glob(os.path.join(mask_dir, "*.png"))
    mask_paths.sort()
    return mask_paths


def get_centroid(mask):
    """计算掩码中非零区域的几何中心（用于条纹指向中心）"""
    non_zero = np.nonzero(mask)
    if len(non_zero[0]) == 0 or len(non_zero[1]) == 0:
        return (mask.shape[0] // 2, mask.shape[1] // 2)  # 无区域时返回图像中心
    centroid_y = int(np.mean(non_zero[0]))
    centroid_x = int(np.mean(non_zero[1]))
    return (centroid_y, centroid_x)


def generate_base_stripe(mask, centroid, target_label=TEMP_LABEL, density1=0.05, branch_level1=2, stripe_step1=3,
                         stripe_width1=2):
    """
    【独立函数】生成第一版基础树枝状条纹掩码（临时标签为10）
    :param mask: 原始掩码
    :param centroid: 目标区域几何中心 (y, x)
    :param target_label: 临时标签（默认10）
    :param density1: 第一版条纹密度（独立参数）
    :param branch_level1: 第一版分支层数（独立参数）
    :param stripe_step1: 第一版延伸步数（独立参数）
    :param stripe_width1: 第一版条纹宽度（独立参数）
    :return: 带临时标签的掩码、基础方向向量(y/x)、距离变换、非零掩码、目标掩码、第一版条纹掩码
    """
    h, w = mask.shape
    # 复制原始掩码，用于绘制临时标签的第一版条纹
    base_stripe_mask = mask.copy()
    target_mask = (mask == TARGET_LABEL).astype(np.uint8)  # 原始目标标签（5）的掩码
    non_zero_mask = (mask > 0).astype(np.uint8)

    # 生成网格坐标
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # 计算基础方向向量（指向中心）
    dir_y_v1 = centroid[0] - y_coords
    dir_x_v1 = centroid[1] - x_coords
    dir_norm_v1 = np.sqrt(dir_y_v1 ** 2 + dir_x_v1 ** 2) + 1e-8
    dir_y_v1 = dir_y_v1 / dir_norm_v1
    dir_x_v1 = dir_x_v1 / dir_norm_v1

    # 第一版分支扰动（独立参数控制）
    np.random.seed(None)
    for _ in range(branch_level1):
        offset_y = np.random.normal(0, 0.5, (h, w)) * target_mask
        offset_x = np.random.normal(0, 0.5, (h, w)) * target_mask
        dir_y_v1 += offset_y
        dir_x_v1 += offset_x

    # 距离变换
    dist_transform = cv2.distanceTransform(target_mask * 255, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    # 第一版随机密度掩码（独立参数）
    random_mask = np.random.rand(h, w) < density1
    random_mask = random_mask.astype(np.uint8) * target_mask * non_zero_mask

    # 第一版条纹加粗偏移（独立宽度）
    offsets1 = []
    for dy in range(-stripe_width1 // 2, stripe_width1 // 2 + 1):
        for dx in range(-stripe_width1 // 2, stripe_width1 // 2 + 1):
            if dy != 0 or dx != 0:
                offsets1.append((dy, dx))

    # 绘制第一版基础条纹：将条纹位置置为临时标签10
    for y in range(h):
        for x in range(w):
            if random_mask[y, x] == 1 and dist_transform[y, x] > 0:
                current_y, current_x = y, x
                for step in range(stripe_step1):
                    if 0 <= current_y < h and 0 <= current_x < w and non_zero_mask[current_y, current_x] == 1:
                        # 置为临时标签10
                        base_stripe_mask[current_y, current_x] = target_label
                        # 第一版加粗：同样置为10
                        for dy, dx in offsets1:
                            ny = current_y + dy
                            nx = current_x + dx
                            if 0 <= ny < h and 0 <= nx < w and non_zero_mask[ny, nx] == 1:
                                base_stripe_mask[ny, nx] = target_label
                    # 沿基础方向更新
                    ny_step = int(current_y + dir_y_v1[current_y, current_x] * 1)
                    nx_step = int(current_x + dir_x_v1[current_y, current_x] * 1)
                    if 0 <= ny_step < h and 0 <= nx_step < w and non_zero_mask[ny_step, nx_step] == 1:
                        current_y, current_x = ny_step, nx_step
                    else:
                        break

    # 生成第一版条纹的掩码（仅label=10的区域）
    v1_stripe_only_mask = (base_stripe_mask == TEMP_LABEL).astype(np.uint8)
    return base_stripe_mask, dir_y_v1, dir_x_v1, dist_transform, non_zero_mask, target_mask, v1_stripe_only_mask


def generate_offset_stripe(base_stripe_mask, centroid, dir_y_v1, dir_x_v1, dist_transform, non_zero_mask, target_mask,
                           v1_stripe_only_mask, density2=0.03, branch_level2=1, stripe_step2=2, stripe_width2=1,
                           offset_angle=15):
    """
    【独立函数】生成第二版带随机偏移的条纹掩码（仅在label=10的区域生成）
    增加空值校验，避免采样失败
    """
    h, w = base_stripe_mask.shape
    # 复制第一版的掩码，用于绘制第二版条纹
    offset_stripe_mask = base_stripe_mask.copy()

    # 提取第一版条纹的像素位置（仅label=10的区域）
    v1_y, v1_x = np.nonzero(v1_stripe_only_mask)
    v1_pixel_count = len(v1_y)

    # 空值校验1：第一版无条纹像素，直接返回原掩码
    if v1_pixel_count == 0:
        print("警告：第一版条纹无像素，跳过第二版条纹生成")
        return offset_stripe_mask

    # 计算第二版需要采样的像素数
    sample_count = int(v1_pixel_count * density2)
    # 空值校验2：采样数为0，直接返回原掩码
    if sample_count == 0:
        return offset_stripe_mask
    # 空值校验3：采样数超过总像素数，调整为允许重复采样或采样全部
    replace = False
    if sample_count > v1_pixel_count:
        replace = True  # 允许重复采样
        # 也可以选择采样全部像素：sample_count = v1_pixel_count

    # 随机采样第一版条纹像素
    keep_idx = np.random.choice(v1_pixel_count, sample_count, replace=replace)
    v1_y = v1_y[keep_idx]
    v1_x = v1_x[keep_idx]

    # 偏移角度转弧度
    offset_rad = np.radians(offset_angle)
    # 为每个基础像素生成随机偏移角度（独立）
    random_angles = np.random.uniform(-offset_rad, offset_rad, len(v1_y))

    # 第二版条纹加粗偏移（独立宽度）
    offsets2 = []
    for dy in range(-stripe_width2 // 2, stripe_width2 // 2 + 1):
        for dx in range(-stripe_width2 // 2, stripe_width2 // 2 + 1):
            if dy != 0 or dx != 0:
                offsets2.append((dy, dx))

    # 第二版分支扰动（独立参数）
    dir_y_v2 = dir_y_v1.copy()
    dir_x_v2 = dir_x_v1.copy()
    np.random.seed(None)
    for _ in range(branch_level2):
        offset_y = np.random.normal(0, 0.3, (h, w)) * target_mask
        offset_x = np.random.normal(0, 0.3, (h, w)) * target_mask
        dir_y_v2 += offset_y
        dir_x_v2 += offset_x

    # 绘制第二版偏移条纹：仅在label=10的区域生成，且生成后仍置为10
    for i in range(len(v1_y)):
        y = v1_y[i]
        x = v1_x[i]
        # 强制校验：仅在label=10且非零的区域生成
        if offset_stripe_mask[y, x] != TEMP_LABEL or non_zero_mask[y, x] != 1 or dist_transform[y, x] <= 0:
            continue

        # 基础方向向量 + 第二版分支扰动
        base_dir_y = dir_y_v2[y, x]
        base_dir_x = dir_y_v2[y, x]  # 修正：原代码笔误为dir_x_v1，此处改为dir_x_v2

        # 随机角度旋转实现方向偏移
        angle = random_angles[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        offset_dir_y = base_dir_y * cos_a - base_dir_x * sin_a
        offset_dir_x = base_dir_y * sin_a + base_dir_x * cos_a

        # 沿偏移方向延伸（第二版独立步数），仅在label=10的区域绘制
        current_y, current_x = y, x
        for step in range(stripe_step2):
            if 0 <= current_y < h and 0 <= current_x < w:
                # 仅在第一版条纹（label=10）的区域绘制第二版条纹
                if offset_stripe_mask[current_y, current_x] == TEMP_LABEL:
                    # 仍置为临时标签10（最终统一替换为5）
                    offset_stripe_mask[current_y, current_x] = TEMP_LABEL
                    # 第二版独立加粗：同样仅在label=10区域绘制
                    for dy, dx in offsets2:
                        ny = current_y + dy
                        nx = current_x + dx
                        if 0 <= ny < h and 0 <= nx < w and offset_stripe_mask[ny, nx] == TEMP_LABEL and non_zero_mask[
                            ny, nx] == 1:
                            offset_stripe_mask[ny, nx] = TEMP_LABEL
            # 沿偏移方向更新位置
            ny_step = int(current_y + offset_dir_y * 1)
            nx_step = int(current_x + offset_dir_x * 1)
            # 新位置也必须是label=10的区域，否则停止延伸
            if 0 <= ny_step < h and 0 <= nx_step < w and offset_stripe_mask[ny_step, nx_step] == TEMP_LABEL and \
                    non_zero_mask[ny_step, nx_step] == 1:
                current_y, current_x = ny_step, nx_step
            else:
                break

    return offset_stripe_mask


def convert_temp_label_to_target(mask, temp_label=TEMP_LABEL, target_label=TARGET_LABEL):
    """将掩码中的临时标签（10）转换为最终目标标签（5）"""
    mask[mask == temp_label] = target_label
    return mask


def inward_fracture_mask(mask,
                         # 第一版条纹独立参数
                         density1=0.05, branch_level1=2, stripe_step1=3, stripe_width1=2,
                         # 第二版条纹独立参数
                         density2=0.03, branch_level2=1, stripe_step2=2, stripe_width2=1, offset_angle=15,
                         # 通用深度参数
                         fracture_depth=20):
    """
    生成双层条纹（第一版label=10，第二版仅在10上生成，最终转5）
    """
    # 1. 原始掩码>0区域约束
    non_zero_mask = (mask > 0).astype(np.uint8)
    if np.sum(non_zero_mask) == 0:
        return mask.copy()

    # 2. 目标标签区域约束（原始label=5）
    target_mask = (mask == TARGET_LABEL).astype(np.uint8)
    if np.sum(target_mask) == 0:
        return mask.copy()

    # 3. 计算目标区域的几何中心
    centroid = get_centroid(target_mask)

    # 4. 生成第一版条纹：置为临时标签10
    base_stripe_mask, dir_y_v1, dir_x_v1, dist_transform, non_zero_mask, target_mask, v1_stripe_only_mask = generate_base_stripe(
        mask, centroid, TEMP_LABEL, density1, branch_level1, stripe_step1, stripe_width1
    )

    # 5. 生成第二版条纹：仅在label=10的区域生成
    offset_stripe_mask = generate_offset_stripe(
        base_stripe_mask, centroid, dir_y_v1, dir_x_v1, dist_transform, non_zero_mask, target_mask,
        v1_stripe_only_mask, density2, branch_level2, stripe_step2, stripe_width2, offset_angle
    )

    # 6. 最终转换：将所有label=10的位置置为5
    final_mask = convert_temp_label_to_target(offset_stripe_mask, TEMP_LABEL, TARGET_LABEL)

    # return final_mask
    return offset_stripe_mask


def process_all_masks():
    """处理所有掩码并保存（两版条纹参数独立调节）"""
    mask_paths = load_mask_files(MASK_DIR)
    if not mask_paths:
        print("未找到任何PNG掩码文件！")
        return

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"跳过无效文件：{mask_path}")
            continue

        filename = os.path.basename(mask_path)
        print(f"处理掩码：{filename}")

        # 调用函数时，独立配置两版条纹的所有参数！
        modified_mask = inward_fracture_mask(
            mask,
            # 第一版基础条纹参数：适当提高密度，避免无像素
            density1=0.005, branch_level1=5, stripe_step1=20, stripe_width1=2,
            # 第二版偏移条纹参数
            density2=0.1, branch_level2=2, stripe_step2=15, stripe_width2=1, offset_angle=20,
            # 通用深度参数
            fracture_depth=30
        )

        # 保存修改后的掩码
        modified_save_path = os.path.join(NEW_MASK_DIR, filename)
        cv2.imwrite(modified_save_path, modified_mask)

        # 保存原始掩码
        original_save_path = os.path.join(NEW_MASK_DIR, f"original_{filename}")
        cv2.imwrite(original_save_path, mask)

    print("所有掩码双层条纹处理完成！（第一版label=10→第二版仅在10生成→最终转5）")


def generate_mask_view():
    """生成可视化掩码（label=5置为255，其余置为0）"""
    mask_paths = load_mask_files(NEW_MASK_DIR)
    if not mask_paths:
        print("new_mask目录中未找到任何PNG掩码文件！")
        return

    for mask_path in mask_paths:
        filename = os.path.basename(mask_path)
        if filename.startswith("original_"):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"跳过无效文件：{mask_path}")
            continue

        print(f"生成可视化掩码：{filename}")
        view_mask = np.zeros_like(mask)
        view_mask[mask == 5] = 255  # 仅显示最终的label=5区域
        view_save_path = os.path.join(MASK_VIEW_DIR, filename)
        cv2.imwrite(view_save_path, view_mask)

    print("可视化掩码生成完成！")


if __name__ == "__main__":
    process_all_masks()
    generate_mask_view()