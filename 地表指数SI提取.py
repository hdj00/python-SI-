import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# === 自定义窗口大小 ===
win_size = 50

# === 读取DEM ===
with rasterio.open("D:\\lunwenfinal\\fill.tif") as src:
    dem = src.read(1).astype(float)
    profile = src.profile

# 替换无效值
dem[dem == src.nodata] = np.nan

# === 定义窗口函数 ===

def hi_func(block):
    block = block[~np.isnan(block)]
    if block.size < 3:
        return np.nan
    return (np.mean(block) - np.min(block)) / (np.max(block) - np.min(block) + 1e-6)

def sr_func(block):
    block = block[~np.isnan(block)]
    if block.size < 3:
        return np.nan
    std = np.std(block)
    return 1/ np.cos(std * np.pi / 180)
    return np.sqrt(1 + np.tan(std * np.pi / 180) ** 2)

# === 应用滑动窗口滤波 ===
print("正在计算 HI...")
hi = generic_filter(dem, hi_func, size=win_size, mode='nearest')

print("正在计算 SR...")
sr = generic_filter(dem, sr_func, size=win_size, mode='nearest')

# === 归一化 DEM, HI, SR ===
def normalize(arr):
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6)

hi_n = normalize(hi)
sr_n = normalize(sr)
dem_n = normalize(dem)

# === 计算 SI ===
print("正在计算 SI...")
si = (hi_n * dem_n) - sr_n


# === 可视化函数 ===
def show_map(data, title, cmap='viridis'):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Value')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === 显示结果 ===
show_map(hi, 'Hypsometric Integral (HI)')
show_map(sr, 'Surface Roughness (SR)')
show_map(si, 'Surface Index (SI)')


def write_geotiff(filename, array, profile, nodata_val=np.nan):
    """将 numpy 数组写入 GeoTIFF 文件"""
    profile_out = profile.copy()
    profile_out.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': nodata_val
    })
    with rasterio.open(filename, 'w', **profile_out) as dst:
        dst.write(array.astype('float32'), 1)
        write_geotiff('HI.tif', hi, profile)

write_geotiff('C:\\Users\\0000\\Desktop\\SI.tif', si, profile)

print("✅ HI, SR, SI 提取完毕！已保存为 GeoTIFF 文件。")
