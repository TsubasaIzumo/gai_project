# CASA Simulation Script based on Table 1 parameters - 修复版 + 详细调试
# 解决dirty image维度问题和单值数组问题
# 兼容旧版CASA (移除不支持的参数)

import os
import numpy as np
import shutil
import uuid
from datetime import datetime
import pickle
import time
import sys
import traceback

# 开始时间戳，用于计算总运行时间
start_time = time.time()

# 激活详细调试模式
DEBUG = True


def debug_print(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs)
        sys.stdout.flush()  # 确保立即显示


# 错误处理函数
def log_error(msg, e=None):
    print("ERROR: " + msg)
    if e:
        print(f"Exception: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    sys.stdout.flush()  # 确保立即显示


# 检查CASA版本 - 轻量级方式
print("\n========== CASA环境检查 ==========")
try:
    # 尝试获取CASA版本信息的多种方式
    casa_version = "未知"
    try:
        import casac

        casa_version = "CASA (casac可用)"
    except:
        pass

    # 仅检查是否可以访问关键任务
    try:
        if 'tclean' in globals():
            casa_version += " - tclean可用"
    except:
        pass

    print(f"CASA状态: {casa_version}")
    print(f"工作目录: {os.getcwd()}")
    print("CASA环境检查完成")
except Exception as e:
    print(f"注意: CASA环境检查跳过: {str(e)}")

# 修改为用户确定有权限的目录
base_dir = os.path.expanduser("~/data/casa_workspace/simulated_data")
print(f"使用基本目录: {base_dir}")

# 创建数据和numpy子目录
data_dir = os.path.join(base_dir, "data")
numpy_dir = os.path.join(base_dir, "numpy_arrays")
true_dir = os.path.join(numpy_dir, "true")
dirty_dir = os.path.join(numpy_dir, "dirty")
log_dir = os.path.join(base_dir, "logs")

# 确保目录存在
for directory in [base_dir, data_dir, numpy_dir, true_dir, dirty_dir, log_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# 创建日志文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"simulation_log_{timestamp}.txt")
print(f"日志将保存至: {log_file}")

# 设置日志记录
import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# 启用日志记录
sys.stdout = Logger(log_file)

print("\n========== 清理之前的模拟结果 ==========")
for directory in [data_dir, numpy_dir, true_dir, dirty_dir]:
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                debug_print(f"删除文件: {item_path}")

# 存储所有模拟的sky_keys和ra_dec信息
all_sky_keys = []
all_ra_dec = {}  # 改为字典格式，与样本一致

# 存储所有射电源信息的字典
all_sources_info = {}

print("\n========== 开始ALMA模拟 ==========")
print("使用Table 1参数配置")

# 设置Table 1参数
field_name = "COSMOS"
field_coords = "J2000 10h00m28.6s +02d12m21.0s"
num_pointings = 9164
num_antennas = 50
central_freq = "230GHz"  # Band 6
channel_width = 7.8  # MHz
num_channels = 240
total_bandwidth = channel_width * num_channels  # MHz
integration_time = "10s"
total_time = "20min"
hour_angle = "transit"

# 为实际目的，限制模拟次数
max_sims = 1000  # 可根据需要调整模拟次数
print(f"计划执行 {max_sims} 次模拟")

# 保存当前工作目录
original_dir = os.getcwd()
debug_print(f"原始工作目录: {original_dir}")

# 计算COSMOS场内合理的坐标范围（约1度范围）
ra_base = 150.1192  # COSMOS 中心RA
dec_base = 2.2058  # COSMOS 中心DEC
debug_print(f"COSMOS场中心坐标: RA={ra_base}, DEC={dec_base}")

# 运行模拟
sim_success = 0
sim_failed = 0

for sim_idx in range(max_sims):
    print(f"\n========== 运行模拟 {sim_idx + 1}/{max_sims} ==========")
    sim_start = time.time()

    try:
        # 项目名称 - 只使用名称，不包含路径
        project_name = f"sim_{timestamp}_{sim_idx}"
        debug_print(f"项目名称: {project_name}")

        # 创建UUID格式的sky_key并保存
        sky_key = str(uuid.uuid4())
        all_sky_keys.append(sky_key)
        debug_print(f"Sky key: {sky_key}")

        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            debug_print(f"创建数据目录: {data_dir}")

        # 创建组件列表路径
        cl_file = os.path.join(data_dir, f"{project_name}.cl")
        debug_print(f"组件列表文件: {cl_file}")

        # 确保组件列表不存在
        if os.path.exists(cl_file):
            os.system(f"rm -rf {cl_file}")
            debug_print(f"删除已存在的组件列表文件: {cl_file}")

        cl.done()  # 清除任何已存在的列表
        debug_print("清除已有组件列表")

        # 为每个模拟生成不同的相位中心，在COSMOS场内合理范围内
        # 生成随机偏移量，范围约±0.5度
        ra_offset = np.random.uniform(-0.6, 0.6)
        dec_offset = np.random.uniform(-0.6, 0.6)

        sim_ra = ra_base + ra_offset
        sim_dec = dec_base + dec_offset

        # 将相位中心转换为CASA格式的坐标字符串
        sim_ra_hms = f"{int(sim_ra / 15)}h{int((sim_ra / 15 % 1) * 60):02d}m{((sim_ra / 15 % 1) * 60 % 1) * 60:.1f}s"
        sim_dec_dms = f"{'+' if sim_dec >= 0 else '-'}{abs(int(sim_dec))}d{int(abs(sim_dec) % 1 * 60):02d}m{(abs(sim_dec) % 1 * 60 % 1) * 60:.1f}s"
        sim_coords = f"J2000 {sim_ra_hms} {sim_dec_dms}"

        # 保存相位中心（用字典格式）
        all_ra_dec[sky_key] = {'RA': sim_ra, 'DEC': sim_dec}

        print(f"使用相位中心: RA={sim_ra}, DEC={sim_dec}")
        debug_print(f"相位中心字符串: {sim_coords}")

        # 随机生成1-5个源，确保至少有1个源
        num_sources = np.random.randint(1, 6)
        sources_info = []

        print(f"生成 {num_sources} 个射电源")

        for src_idx in range(num_sources):
            # 生成随机源属性 - 遵循表格中的参数
            # 流量范围: 0.05-0.5 mJy (从表格中获取)
            flux = np.random.uniform(0.05, 0.5) / 1000.0  # 0.05-0.5 mJy 转换为 Jy
            # 轴长范围: 0.4"-0.8" (从表格中获取)
            major_axis = np.random.uniform(0.4, 0.8)  # arcsec
            minor_axis = np.random.uniform(0.4, major_axis)  # arcsec
            pa = np.random.uniform(0, 360)  # degrees

            # 随机位置在主波束内 (22.86" 从表格中获取)
            radius_arcsec = np.random.uniform(0, 22.86 / 2)
            angle_rad = np.random.uniform(0, 2 * np.pi)

            # 转换为RA/DEC偏移
            ra_src_offset = radius_arcsec * np.cos(angle_rad) / 3600.0  # 转换arcsec到度
            dec_src_offset = radius_arcsec * np.sin(angle_rad) / 3600.0

            # 计算源的实际RA和DEC
            ra = sim_ra + ra_src_offset
            dec = sim_dec + dec_src_offset

            # 计算SNR（这里是简化估计）
            # 使用表格中的RMS噪声水平 ~50μJy/beam
            noise_level = 50e-6  # Jy/beam (50μJy)
            snr = flux / noise_level

            # 归一化SNR（考虑源的大小）
            beam_maj = 0.89  # arcsec - 表格中的合成波束大小
            beam_min = 0.82  # arcsec
            snr_normalized = snr * (beam_maj * beam_min) / np.sqrt(
                (beam_maj ** 2 + major_axis ** 2) * (beam_min ** 2 + minor_axis ** 2))

            # 修正：为每个源创建正确的坐标字符串
            ra_src_hms = f"{int(ra / 15)}h{int((ra / 15 % 1) * 60):02d}m{((ra / 15 % 1) * 60 % 1) * 60:.1f}s"
            dec_src_dms = f"{'+' if dec >= 0 else '-'}{abs(int(dec))}d{int(abs(dec) % 1 * 60):02d}m{(abs(dec) % 1 * 60 % 1) * 60:.1f}s"
            src_coords = f"J2000 {ra_src_hms} {dec_src_dms}"

            # 使用源的实际坐标
            debug_print(f"源 {src_idx + 1} 坐标: {src_coords}, 流量: {flux} Jy")
            debug_print(f"源 {src_idx + 1} 属性: 长轴={major_axis}arcsec, 短轴={minor_axis}arcsec, PA={pa}度")

            cl.addcomponent(
                dir=src_coords,
                flux=flux,
                fluxunit="Jy",
                freq=central_freq,
                shape="gaussian",
                majoraxis=f"{major_axis}arcsec",
                minoraxis=f"{minor_axis}arcsec",
                positionangle=f"{pa}deg",
                spectrumtype="spectral index",
                index=0  # 表格中的谱指数为0
            )

            print(f"添加源于 {src_coords} 流量 {flux} Jy")

            sources_info.append([ra, dec, snr, snr_normalized, flux, major_axis, minor_axis])

        # 保存组件列表
        cl.rename(cl_file)
        cl.done()
        debug_print(f"组件列表已保存到 {cl_file}")

        # 保存射电源信息
        all_sources_info[sky_key] = sources_info

        # 切换到数据目录
        os.chdir(data_dir)
        debug_print(f"切换工作目录到 {data_dir}")

        # 使用更合适的频谱设置进行模拟 - 使用模拟特定的相位中心
        print("\n----- 执行simobserve -----")
        simobserve(
            project=project_name,
            skymodel="",
            complist=cl_file,
            setpointings=True,
            integration=integration_time,
            direction=sim_coords,  # 使用模拟特定的相位中心
            mapsize="22.86arcsec",  # 表格中的主波束大小
            obsmode="int",
            totaltime=total_time,
            antennalist="alma.cycle5.3.cfg",
            hourangle=hour_angle,
            thermalnoise="tsys-atm",
            user_pwv=1.796,  # 表格中的PWV值
            graphics="none",
            incenter=central_freq,
            inwidth=f"{total_bandwidth}MHz",
            incell="0.1arcsec",  # 表格中的像素大小
            overwrite=True
        )
        debug_print("simobserve完成")

        # simobserve创建的项目目录和MS文件路径
        project_dir = os.path.join(data_dir, project_name)
        ms_file = os.path.join(project_dir, f"{project_name}.alma.cycle5.3.ms")

        # 检查MS文件
        if not os.path.exists(ms_file):
            raise FileNotFoundError(f"MS文件未创建: {ms_file}")

        ms_size = os.path.getsize(ms_file)
        debug_print(f"MS文件大小: {ms_size} 字节")

        # 显示MS文件的基本信息
        print("\n----- MS文件信息 -----")
        try:
            listobs(vis=ms_file)
        except Exception as e:
            log_error("listobs失败", e)

        # 统计是否有足够的数据点
        try:
            ms.open(ms_file)
            ms_data = ms.getdata(['data'])
            data_shape = ms_data['data'].shape
            print(f"MS数据形状: {data_shape}")
            data_points = np.prod(data_shape)
            print(f"数据点总数: {data_points}")
            ms.close()
        except Exception as e:
            log_error("无法获取MS数据统计", e)

        # 首先生成dirty image（不进行清洁）
        print("\n----- 生成Dirty Image (不清洁) -----")
        dirty_imagename = os.path.join(project_dir, "dirty")
        debug_print(f"Dirty image将保存到: {dirty_imagename}")

        tclean(
            vis=ms_file,
            imagename=dirty_imagename,
            imsize=[512, 512],  # 表格中的sky模型维度
            cell="0.1arcsec",  # 表格中的像素大小
            specmode="mfs",
            weighting="natural",  # 表格中的加权方式
            robust=0.5,  # 表格中的robust参数
            niter=0,  # 设置为0，只创建dirty image不进行清洁
            interactive=False,
            savemodel='none',  # 不保存模型
            gridder='standard'  # 使用标准网格器
        )
        debug_print("tclean (dirty) 完成")

        # 导出dirty image为FITS
        dirty_image_file = os.path.join(project_dir, "dirty.image")
        dirty_fits_file = os.path.join(project_dir, "dirty.fits")

        if not os.path.exists(dirty_image_file):
            raise FileNotFoundError(f"Dirty image文件未创建: {dirty_image_file}")

        # 检查dirty image文件大小
        dirty_size = os.path.getsize(dirty_image_file)
        print(f"Dirty image文件大小: {dirty_size} 字节")

        # 检查dirty image文件是否为空
        if dirty_size < 1000:
            print("警告: Dirty image文件异常小!")

        exportfits(
            imagename=dirty_image_file,
            fitsimage=dirty_fits_file,
            overwrite=True
        )
        debug_print(f"导出dirty image到FITS: {dirty_fits_file}")

        # 进行清洁成像
        print("\n----- 执行Clean成像 -----")
        clean_imagename = os.path.join(project_dir, "clean")
        debug_print(f"Clean image将保存到: {clean_imagename}")

        tclean(
            vis=ms_file,
            imagename=clean_imagename,
            imsize=[512, 512],  # 表格中的sky模型维度
            cell="0.1arcsec",  # 表格中的像素大小
            specmode="mfs",
            deconvolver="hogbom",
            weighting="natural",  # 表格中的加权方式
            robust=0.5,  # 表格中的robust参数
            niter=1000,
            threshold="50uJy",  # 表格中的RMS噪声水平
            interactive=False,
            savemodel='modelcolumn'  # 保存模型到MS文件
        )
        debug_print("tclean (clean) 完成")

        # 导出clean.image为FITS
        image_file = os.path.join(project_dir, "clean.image")
        fits_file = os.path.join(project_dir, "clean.fits")

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Clean image文件未创建: {image_file}")

        # 检查clean image文件大小
        clean_size = os.path.getsize(image_file)
        print(f"Clean image文件大小: {clean_size} 字节")

        exportfits(
            imagename=image_file,
            fitsimage=fits_file,
            overwrite=True
        )
        debug_print(f"导出clean image到FITS: {fits_file}")

        # ========== 使用CASA原生工具处理图像 ==========
        print("\n----- 从CASA图像创建NumPy数组 -----")

        # 确保CASA图像工具是可用的
        ia.done()  # 关闭之前可能打开的图像

        # 1. 处理clean图像
        clean_image_file = os.path.join(project_dir, "clean.image")
        if not os.path.exists(clean_image_file):
            raise FileNotFoundError(f"Clean image文件不存在: {clean_image_file}")

        print(f"正在读取clean图像: {clean_image_file}")
        ia.open(clean_image_file)
        clean_image_2d = ia.getchunk()
        # 如果图像有多个通道和偏振，只保留第一个通道和偏振
        if len(clean_image_2d.shape) > 2:
            print(f"原始clean图像形状: {clean_image_2d.shape}, 提取第一个通道和偏振")
            clean_image_2d = clean_image_2d[:, :, 0, 0]
        ia.close()

        # 2. 处理dirty图像
        dirty_image_file = os.path.join(project_dir, "dirty.image")
        if not os.path.exists(dirty_image_file):
            raise FileNotFoundError(f"Dirty image文件不存在: {dirty_image_file}")

        print(f"正在读取dirty图像: {dirty_image_file}")
        ia.open(dirty_image_file)
        dirty_image_2d = ia.getchunk()
        # 如果图像有多个通道和偏振，只保留第一个通道和偏振
        if len(dirty_image_2d.shape) > 2:
            print(f"原始dirty图像形状: {dirty_image_2d.shape}, 提取第一个通道和偏振")
            dirty_image_2d = dirty_image_2d[:, :, 0, 0]
        ia.close()

        print("CASA图像成功转换为NumPy数组")

        # 3. 应用圆形掩膜 - 使结果与参考图像一致
        print("应用圆形掩膜...")
        y, x = np.ogrid[:512, :512]
        center = 512/2
        mask = (x - center)**2 + (y - center)**2 <= center**2

        # 复制数组后应用掩膜
        clean_image_masked = np.copy(clean_image_2d)
        dirty_image_masked = np.copy(dirty_image_2d)

        # 掩膜外区域设置为NaN
        clean_image_masked[~mask] = np.nan
        dirty_image_masked[~mask] = np.nan

        print(f"Clean image形状: {clean_image_masked.shape}")
        print(f"Clean image统计: 最小={np.nanmin(clean_image_masked):.3e}, 最大={np.nanmax(clean_image_masked):.3e}, 平均={np.nanmean(clean_image_masked):.3e}")
        print(f"Dirty image形状: {dirty_image_masked.shape}")
        print(f"Dirty image统计: 最小={np.nanmin(dirty_image_masked):.3e}, 最大={np.nanmax(dirty_image_masked):.3e}, 平均={np.nanmean(dirty_image_masked):.3e}")

        # 保存clean image (true)
        clean_file = os.path.join(true_dir, f"{sim_idx:05d}.npy")
        np.save(clean_file, clean_image_masked)
        print(f"保存Clean image到: {clean_file}")

        # 保存dirty image
        dirty_file = os.path.join(dirty_dir, f"{sim_idx:05d}.npy")
        np.save(dirty_file, dirty_image_masked)
        print(f"保存Dirty image到: {dirty_file}")

        # 检查文件大小
        clean_npy_size = os.path.getsize(clean_file)
        dirty_npy_size = os.path.getsize(dirty_file)
        print(f"Clean NPY文件大小: {clean_npy_size} 字节")
        print(f"Dirty NPY文件大小: {dirty_npy_size} 字节")

        # 最终验证文件
        try:
            print("\n----- 验证保存的文件 -----")
            test_clean = np.load(clean_file)
            print(f"读取Clean image: 形状={test_clean.shape}, 最小={np.nanmin(test_clean)}, 最大={np.nanmax(test_clean)}")

            test_dirty = np.load(dirty_file)
            print(f"读取Dirty image: 形状={test_dirty.shape}, 最小={np.nanmin(test_dirty)}, 最大={np.nanmax(test_dirty)}")

        except Exception as e:
            log_error("验证保存的文件时出错", e)

        sim_end = time.time()
        sim_duration = sim_end - sim_start
        print(f"\n模拟 {sim_idx + 1} 完成，用时 {sim_duration:.2f} 秒")
        sim_success += 1

    except Exception as e:
        sim_end = time.time()
        sim_duration = sim_end - sim_start
        log_error(f"模拟 {sim_idx + 1} 失败，用时 {sim_duration:.2f} 秒", e)
        sim_failed += 1

# 保存sky_keys.npy
print("\n========== 保存最终结果 ==========")
np.save(os.path.join(numpy_dir, "sky_keys.npy"), np.array(all_sky_keys))
print(f"保存sky_keys.npy，包含 {len(all_sky_keys)} 个keys")

# 保存ra_dec.npy - 保存为字典，与样本一致
with open(os.path.join(numpy_dir, "ra_dec.npy"), 'wb') as f:
    pickle.dump(all_ra_dec, f)
print(f"保存ra_dec.npy，包含 {len(all_ra_dec)} 个条目")

# 保存射电源信息
with open(os.path.join(numpy_dir, "sky_sources_snr_extended.npy"), 'wb') as f:
    pickle.dump(all_sources_info, f)
print(f"保存sky_sources_snr_extended.npy，包含 {len(all_sources_info)} 个条目")

# 恢复原始工作目录
os.chdir(original_dir)
print(f"恢复工作目录到 {original_dir}")

# 总结
end_time = time.time()
total_time = end_time - start_time
print("\n========== 模拟总结 ==========")
print(f"总运行时间: {total_time:.2f} 秒")
print(f"成功模拟: {sim_success}/{max_sims}")
print(f"失败模拟: {sim_failed}/{max_sims}")
print(f"输出目录: {base_dir}")
print(f"数据文件位于: {data_dir}")
print(f"Clean images (true) 位于: {true_dir}")
print(f"Dirty images 位于: {dirty_dir}")
print(f"其他NumPy数组位于: {numpy_dir}")
print(f"日志文件: {log_file}")
print("========== 模拟完成 ==========")

# 关闭日志文件
if isinstance(sys.stdout, Logger):
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal