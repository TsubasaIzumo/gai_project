# CASA Simulation Script aligned to DDPM training data approach
# - 以TCLEAN (niter=0) 生成dirty image作为条件输入
# - Band-6 (230 GHz), 240 channels @ 7.8 MHz (MFS成像)
# - COSMOS中心周围 1 度半径内相位中心随机采样
# - 512x512 @ 0.1"/pix → 51.2" 视场

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

# 固定随机种子以保证可重复性
RNG_SEED = 20251111
np.random.seed(RNG_SEED)


# 错误处理函数
def log_error(msg, e=None):
    print("ERROR: " + msg)
    if e:
        print(f"Exception: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    sys.stdout.flush()


# 检查CASA版本 - 轻量级方式
print("\n========== CASA环境检查 ==========")
try:
    # 尝试获取CASA版本信息
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
except Exception as e:
    print(f"注意: CASA环境检查跳过: {str(e)}")

# 修改为用户确定有权限的目录
base_dir = os.path.expanduser("~/data/casa_workspace/simulated_data")
print(f"使用基本目录: {base_dir}")

# 创建最小必要的目录
numpy_dir = os.path.join(base_dir, "numpy_arrays")
true_dir = os.path.join(numpy_dir, "true")
dirty_dir = os.path.join(numpy_dir, "dirty")
temp_dir = os.path.join(base_dir, "temp")  # 临时工作目录
log_dir = os.path.join(base_dir, "logs")
checkpoint_dir = os.path.join(base_dir, "checkpoints")

# 确保目录存在
for directory in [base_dir, numpy_dir, true_dir, dirty_dir, temp_dir, log_dir, checkpoint_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# 创建日志文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"simulation_log_{timestamp}.txt")
print(f"日志将保存至: {log_file}")


# 设置日志记录
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


# 清理临时目录
def cleanup_temp_dir():
    """清理临时目录中的所有文件"""
    if os.path.exists(temp_dir):
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"清理文件失败: {item_path} - {str(e)}")
    print("临时目录已清理")


# 断点重连功能：检查是否存在检查点
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
if os.path.exists(checkpoint_file):
    print("\n========== 发现检查点，尝试恢复状态 ==========")
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        all_sky_keys = checkpoint_data.get('all_sky_keys', [])
        all_ra_dec = checkpoint_data.get('all_ra_dec', {})
        all_sources_info = checkpoint_data.get('all_sources_info', {})
        start_sim_idx = checkpoint_data.get('next_sim_idx', 0)
        sim_success = checkpoint_data.get('sim_success', 0)
        sim_failed = checkpoint_data.get('sim_failed', 0)

        print(f"从检查点恢复: 已完成 {start_sim_idx} 次模拟，成功 {sim_success}，失败 {sim_failed}")
        print(
            f"已存在 {len(all_sky_keys)} 个sky_keys，{len(all_ra_dec)} 个ra_dec条目，{len(all_sources_info)} 个sources_info条目")
    except Exception as e:
        print(f"无法加载检查点，将从头开始: {str(e)}")
        all_sky_keys = []
        all_ra_dec = {}
        all_sources_info = {}
        start_sim_idx = 0
        sim_success = 0
        sim_failed = 0
else:
    print("\n========== 未找到检查点，从头开始 ==========")
    all_sky_keys = []
    all_ra_dec = {}
    all_sources_info = {}
    start_sim_idx = 0
    sim_success = 0
    sim_failed = 0

# 清理临时目录
cleanup_temp_dir()

# 指向数优先复现实验规模
num_pointings_with_sources = 9164
target_total_sources = 27632
num_blank_images = 1000

# 计算每幅图的源数分配（≥1）并使总和精确等于 target_total_sources
lambda_per_image = target_total_sources / num_pointings_with_sources
initial_sources_per_image = np.random.poisson(lam=lambda_per_image, size=num_pointings_with_sources)
# 将0重采样为≥1
zeros_idx = np.where(initial_sources_per_image == 0)[0]
if zeros_idx.size > 0:
    initial_sources_per_image[zeros_idx] = 1

current_sum = int(np.sum(initial_sources_per_image))
delta = target_total_sources - current_sum

if delta > 0:
    # 需要增加 delta 个源：随机选择 delta 个指向并 +1
    inc_idx = np.random.choice(num_pointings_with_sources, size=delta, replace=True)
    for idx in inc_idx:
        initial_sources_per_image[idx] += 1
elif delta < 0:
    # 需要减少 -delta 个源：在值>1的指向上 -1（必要时允许多次循环）
    delta = -delta
    dec_candidates = np.where(initial_sources_per_image > 1)[0].tolist()
    while delta > 0 and dec_candidates:
        idx = np.random.choice(dec_candidates)
        initial_sources_per_image[idx] -= 1
        delta -= 1
        if initial_sources_per_image[idx] == 1:
            dec_candidates.remove(idx)

# 再次确认总和
assert int(np.sum(initial_sources_per_image)) == target_total_sources, "源数配额未精确匹配目标总源数"

# 构造最终每次模拟的源数计划：前 9164 为含源，后 1000 为无源
per_sim_num_sources = initial_sources_per_image.tolist() + [0] * num_blank_images
max_sims = len(per_sim_num_sources)
print(f"计划执行 {max_sims} 次模拟（含源 {num_pointings_with_sources}，无源 {num_blank_images}）")

# 保存当前工作目录
original_dir = os.getcwd()

# 计算COSMOS场中心（用于1度范围采样）
ra_base = 150.1192  # COSMOS 中心RA
dec_base = 2.2058  # COSMOS 中心DEC

# 设置模拟参数
field_name = "COSMOS"
field_coords = "J2000 10h00m28.6s +02d12m21.0s"
central_freq = "230GHz"  # Band 6
channel_width = 7.8  # MHz
num_channels = 240
total_bandwidth = channel_width * num_channels  # MHz
integration_time = "10s"
total_time = "20min"
hour_angle = "transit"
primary_beam_arcsec = 22.86  # 主波束直径（用于源位置范围）
cell_arcsec = 0.1
imsize_pixels = 512
fov_arcsec = imsize_pixels * cell_arcsec  # 51.2"


# 函数：保存检查点
def save_checkpoint(idx, sky_keys, ra_dec_dict, sources_info, success, failed):
    checkpoint_data = {
        'all_sky_keys': sky_keys,
        'all_ra_dec': ra_dec_dict,
        'all_sources_info': sources_info,
        'next_sim_idx': idx,
        'sim_success': success,
        'sim_failed': failed,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # 保存检查点
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # 同时保存当前的输出文件
    np.save(os.path.join(numpy_dir, "sky_keys.npy"), np.array(sky_keys))
    with open(os.path.join(numpy_dir, "ra_dec.npy"), 'wb') as f:
        pickle.dump(ra_dec_dict, f)
    with open(os.path.join(numpy_dir, "sky_sources_snr_extended.npy"), 'wb') as f:
        pickle.dump(sources_info, f)

    print(f"\n========== 已保存检查点 {idx}/{max_sims} ==========")
    print(f"成功: {success}, 失败: {failed}, 数据时间: {checkpoint_data['timestamp']}")


# 运行模拟
print("\n========== 开始ALMA模拟 ==========")
print(f"从第 {start_sim_idx + 1}/{max_sims} 个模拟开始")

for sim_idx in range(start_sim_idx, max_sims):
    print(f"\n========== 运行模拟 {sim_idx + 1}/{max_sims} ==========")
    sim_start = time.time()

    try:
        # 确保临时目录清理干净
        cleanup_temp_dir()

        # 切换到临时目录
        os.chdir(temp_dir)

        planned_num_sources = int(per_sim_num_sources[sim_idx])

        # 项目名称
        project_name = f"sim_{timestamp}_{sim_idx}"

        # 创建UUID格式的sky_key并保存
        sky_key = str(uuid.uuid4())
        all_sky_keys.append(sky_key)

        # 创建组件列表路径
        cl_file = os.path.join(temp_dir, f"{project_name}.cl")

        # 确保组件列表不存在
        if os.path.exists(cl_file):
            os.remove(cl_file)

        cl.done()  # 清除任何已存在的列表

        # 为每个模拟生成不同的相位中心：在COSMOS中心1°半径内极坐标均匀采样
        radius_deg = np.sqrt(np.random.uniform(0.0, 1.0)) * 1.0  # 均匀面积采样，最大1度
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        dec_scale = np.cos(np.deg2rad(dec_base))
        ra_offset = (radius_deg * np.cos(theta)) / max(dec_scale, 1e-6)
        dec_offset = radius_deg * np.sin(theta)

        sim_ra = ra_base + ra_offset
        sim_dec = dec_base + dec_offset

        # 将相位中心转换为CASA格式的坐标字符串
        sim_ra_hms = f"{int(sim_ra / 15)}h{int((sim_ra / 15 % 1) * 60):02d}m{((sim_ra / 15 % 1) * 60 % 1) * 60:.1f}s"
        sim_dec_dms = f"{'+' if sim_dec >= 0 else '-'}{abs(int(sim_dec))}d{int(abs(sim_dec) % 1 * 60):02d}m{(abs(sim_dec) % 1 * 60 % 1) * 60:.1f}s"
        sim_coords = f"J2000 {sim_ra_hms} {sim_dec_dms}"

        # 保存相位中心（用字典格式）
        all_ra_dec[sky_key] = {'RA': sim_ra, 'DEC': sim_dec}

        print(f"使用相位中心: RA={sim_ra}, DEC={sim_dec}")

        # 按计划生成源数量
        num_sources = planned_num_sources
        sources_info = []

        print(f"生成 {num_sources} 个射电源")

        for src_idx in range(num_sources):
            # 生成随机源属性
            flux = np.random.uniform(0.05, 0.5) / 1000.0  # 0.05-0.5 mJy → Jy
            major_axis = np.random.uniform(0.4, 0.8)  # arcsec
            minor_axis = np.random.uniform(0.4, major_axis)  # arcsec
            pa = np.random.uniform(0, 360)  # degrees

            radius_arcsec = np.random.uniform(0, primary_beam_arcsec / 2)
            angle_rad = np.random.uniform(0, 2 * np.pi)

            ra_src_offset = radius_arcsec * np.cos(angle_rad) / 3600.0  # arcsec → deg
            dec_src_offset = radius_arcsec * np.sin(angle_rad) / 3600.0

            ra = sim_ra + ra_src_offset
            dec = sim_dec + dec_src_offset

            noise_level = 50e-6  # Jy/beam
            snr = flux / noise_level

            beam_maj = 0.89  # arcsec（示意）
            beam_min = 0.82  # arcsec
            snr_normalized = snr * (beam_maj * beam_min) / np.sqrt(
                (beam_maj ** 2 + major_axis ** 2) * (beam_min ** 2 + minor_axis ** 2))

            ra_src_hms = f"{int(ra / 15)}h{int((ra / 15 % 1) * 60):02d}m{((ra / 15 % 1) * 60 % 1) * 60:.1f}s"
            dec_src_dms = f"{'+' if dec >= 0 else '-'}{abs(int(dec))}d{int(abs(dec) % 1 * 60):02d}m{(abs(dec) % 1 * 60 % 1) * 60:.1f}s"
            src_coords = f"J2000 {ra_src_hms} {dec_src_dms}"

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
                index=0
            )

            print(f"添加源于 {src_coords} 流量 {flux} Jy")

            sources_info.append([ra, dec, snr, snr_normalized, flux, major_axis, minor_axis])

        # 无源样本：添加极弱虚拟源以保证simobserve运行
        if num_sources == 0:
            cl.addcomponent(
                dir=sim_coords,
                flux=1e-12,
                fluxunit="Jy",
                freq=central_freq,
                shape="point",
                spectrumtype="spectral index",
                index=0
            )
            print("无源样本：添加极弱虚拟源以保证simobserve运行")

        cl.rename(cl_file)
        cl.done()

        # 保存射电源信息
        all_sources_info[sky_key] = sources_info

        # 执行simobserve
        print("\n----- 执行simobserve -----")
        simobserve(
            project=project_name,
            skymodel="",
            complist=cl_file,
            setpointings=True,
            integration=integration_time,
            direction=sim_coords,
            mapsize=f"{fov_arcsec}arcsec",
            obsmode="int",
            totaltime=total_time,
            antennalist="alma.cycle5.3.cfg",
            hourangle=hour_angle,
            thermalnoise="tsys-atm",
            user_pwv=1.796,
            graphics="none",
            incenter=central_freq,
            inwidth=f"{total_bandwidth}MHz",
            incell=f"{cell_arcsec}arcsec",
            overwrite=True
        )

        # 获取MS文件路径
        project_dir = os.path.join(temp_dir, project_name)
        ms_file = os.path.join(project_dir, f"{project_name}.alma.cycle5.3.ms")

        if not os.path.exists(ms_file):
            raise FileNotFoundError(f"MS文件未创建: {ms_file}")

        # 生成Dirty Image
        print("\n----- 生成Dirty Image (不清洁) -----")
        dirty_imagename = os.path.join(project_dir, "dirty")

        tclean(
            vis=ms_file,
            imagename=dirty_imagename,
            imsize=[imsize_pixels, imsize_pixels],
            cell=f"{cell_arcsec}arcsec",
            specmode="mfs",
            weighting="natural",
            niter=0,
            interactive=False,
            savemodel='none',
            gridder='standard'
        )

        # 执行Clean成像
        print("\n----- 执行Clean成像 -----")
        clean_imagename = os.path.join(project_dir, "clean")

        tclean(
            vis=ms_file,
            imagename=clean_imagename,
            imsize=[imsize_pixels, imsize_pixels],
            cell=f"{cell_arcsec}arcsec",
            specmode="mfs",
            deconvolver="hogbom",
            weighting="natural",
            niter=1000,
            threshold="50uJy",
            interactive=False,
            savemodel='modelcolumn'
        )

        # 从CASA图像创建NumPy数组并直接保存到最终目录
        print("\n----- 从CASA图像创建NumPy数组 -----")
        ia.done()  # 关闭之前可能打开的图像

        # 处理clean图像
        clean_image_file = os.path.join(project_dir, "clean.image")
        if not os.path.exists(clean_image_file):
            raise FileNotFoundError(f"Clean image文件未创建: {clean_image_file}")

        ia.open(clean_image_file)
        clean_image_2d = ia.getchunk()
        if len(clean_image_2d.shape) > 2:
            clean_image_2d = clean_image_2d[:, :, 0, 0]
        ia.close()

        # 处理dirty图像
        dirty_image_file = os.path.join(project_dir, "dirty.image")
        if not os.path.exists(dirty_image_file):
            raise FileNotFoundError(f"Dirty image文件未创建: {dirty_image_file}")

        ia.open(dirty_image_file)
        dirty_image_2d = ia.getchunk()
        if len(dirty_image_2d.shape) > 2:
            dirty_image_2d = dirty_image_2d[:, :, 0, 0]
        ia.close()

        # 应用圆形掩膜
        y, x = np.ogrid[:imsize_pixels, :imsize_pixels]
        center = imsize_pixels / 2
        mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2

        # 复制数组后应用掩膜
        clean_image_masked = np.copy(clean_image_2d)
        dirty_image_masked = np.copy(dirty_image_2d)

        # 掩膜外区域设置为NaN
        clean_image_masked[~mask] = np.nan
        dirty_image_masked[~mask] = np.nan

        print(f"Clean image形状: {clean_image_masked.shape}")
        print(f"Dirty image形状: {dirty_image_masked.shape}")

        # 保存clean image (true)
        clean_file = os.path.join(true_dir, f"{sim_idx:05d}.npy")
        np.save(clean_file, clean_image_masked)
        print(f"保存Clean image到: {clean_file}")

        # 保存dirty image
        dirty_file = os.path.join(dirty_dir, f"{sim_idx:05d}.npy")
        np.save(dirty_file, dirty_image_masked)
        print(f"保存Dirty image到: {dirty_file}")

        # 清理本次模拟的临时文件
        cleanup_temp_dir()

        sim_end = time.time()
        sim_duration = sim_end - sim_start
        print(f"\n模拟 {sim_idx + 1} 完成，用时 {sim_duration:.2f} 秒")
        sim_success += 1

        # 每500次模拟保存一次检查点
        if (sim_idx + 1) % 500 == 0 or sim_idx == max_sims - 1:
            save_checkpoint(sim_idx + 1, all_sky_keys, all_ra_dec, all_sources_info, sim_success, sim_failed)

    except Exception as e:
        sim_end = time.time()
        sim_duration = sim_end - sim_start
        log_error(f"模拟 {sim_idx + 1} 失败，用时 {sim_duration:.2f} 秒", e)
        sim_failed += 1

        # 失败后清理临时文件
        try:
            cleanup_temp_dir()
        except:
            pass

        # 失败后也保存检查点
        if (sim_idx + 1) % 500 == 0:
            save_checkpoint(sim_idx + 1, all_sky_keys, all_ra_dec, all_sources_info, sim_success, sim_failed)

# 恢复原始工作目录
os.chdir(original_dir)

# 保存最终结果
print("\n========== 保存最终结果 ==========")
np.save(os.path.join(numpy_dir, "sky_keys.npy"), np.array(all_sky_keys))
print(f"保存sky_keys.npy，包含 {len(all_sky_keys)} 个keys")

with open(os.path.join(numpy_dir, "ra_dec.npy"), 'wb') as f:
    pickle.dump(all_ra_dec, f)
print(f"保存ra_dec.npy，包含 {len(all_ra_dec)} 个条目")

with open(os.path.join(numpy_dir, "sky_sources_snr_extended.npy"), 'wb') as f:
    pickle.dump(all_sources_info, f)
print(f"保存sky_sources_snr_extended.npy，包含 {len(all_sources_info)} 个条目")

# 总结
end_time = time.time()
total_time = end_time - start_time
print("\n========== 模拟总结 ==========")
print(f"总运行时间: {total_time:.2f} 秒")
print(f"成功模拟: {sim_success}/{max_sims}")
print(f"失败模拟: {sim_failed}/{max_sims}")
print(f"输出目录: {base_dir}")
print("========== 模拟完成 ==========")

# 关闭日志文件
if isinstance(sys.stdout, Logger):
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal