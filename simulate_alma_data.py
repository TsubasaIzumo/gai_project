# CASA Simulation Script based on Table 1 parameters

import os
import numpy as np
import shutil
from datetime import datetime

# 修改为用户确定有权限的目录
base_dir = os.path.expanduser("~/data/casa_workspace/simulated_data")
print(f"Using base directory: {base_dir}")

# 创建数据和numpy子目录
data_dir = os.path.join(base_dir, "data")
numpy_dir = os.path.join(base_dir, "numpy_arrays")

# 确保目录存在
for directory in [base_dir, data_dir, numpy_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# 清理之前的运行结果，避免文件冲突
print("Cleaning previous simulation results...")
for directory in [data_dir, numpy_dir]:
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)

print("Starting ALMA simulation based on Table 1 parameters")

# Set parameters from Table 1
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

# For practical purposes, limit the number of simulations
max_sims = 10

# 添加时间戳，确保文件名唯一
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存当前工作目录
original_dir = os.getcwd()

# Run simulations
for sim_idx in range(max_sims):
    print(f"Running simulation {sim_idx + 1}/{max_sims}")

    # Project name - 只使用名称，不包含路径
    project_name = f"sim_{timestamp}_{sim_idx}"

    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 创建组件列表路径
    cl_file = os.path.join(data_dir, f"{project_name}.cl")

    # 确保组件列表不存在
    if os.path.exists(cl_file):
        os.system(f"rm -rf {cl_file}")

    cl.done()  # Clean up any existing list

    # 随机生成1-5个源，确保至少有1个源
    num_sources = np.random.randint(1, 6)

    for src_idx in range(num_sources):
        # Generate random source properties
        flux = np.random.uniform(0.05, 0.5) / 1000.0  # mJy to Jy
        major_axis = np.random.uniform(0.4, 0.8)  # arcsec
        minor_axis = np.random.uniform(0.4, major_axis)  # arcsec
        pa = np.random.uniform(0, 360)  # degrees

        # Random position within primary beam
        radius_arcsec = np.random.uniform(0, 22.86 / 2)
        angle_rad = np.random.uniform(0, 2 * np.pi)

        # Add component to the model
        cl.addcomponent(
            dir=field_coords,
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

    # Save component list
    cl.rename(cl_file)
    cl.done()

    # Generate random pointing within 1 degree of field center
    pointing_radius = np.random.uniform(0, 1.0)
    pointing_angle = np.random.uniform(0, 2 * np.pi)
    ra_offset = pointing_radius * np.cos(pointing_angle)
    dec_offset = pointing_radius * np.sin(pointing_angle)

    # 切换到数据目录
    os.chdir(data_dir)

    # 使用更合适的频谱设置进行模拟 - 注意只使用项目名称
    simobserve(
        project=project_name,  # 只使用项目名称，不包含路径
        skymodel="",
        complist=cl_file,  # 使用完整路径指定组件列表
        setpointings=True,
        integration=integration_time,
        direction=field_coords,
        mapsize="22.86arcsec",
        obsmode="int",
        totaltime=total_time,
        antennalist="alma.cycle5.3.cfg",
        hourangle=hour_angle,
        thermalnoise="tsys-atm",
        user_pwv=1.796,
        graphics="none",
        # 以下参数确保频道宽度正确
        incenter=central_freq,
        inwidth=f"{total_bandwidth}MHz",  # 总带宽
        incell="0.1arcsec",
        inbright="0.5mJy/beam",
        overwrite=True  # 添加覆盖选项
    )

    # simobserve创建的项目目录和MS文件路径
    project_dir = os.path.join(data_dir, project_name)  # 项目目录的完整路径
    ms_file = os.path.join(project_dir, f"{project_name}.alma.cycle5.3.ms")

    # 用更匹配的频谱设置进行成像
    tclean(
        vis=ms_file,  # 使用MS文件的完整路径
        imagename=os.path.join(project_dir, "clean"),
        imsize=[512, 512],
        cell="0.1arcsec",
        specmode="mfs",  # 使用多频率合成而不是cube模式
        deconvolver="hogbom",
        weighting="natural",
        robust=0.5,
        niter=1000,
        threshold="50uJy",
        interactive=False
    )

    # 图像和FITS文件的完整路径
    image_file = os.path.join(project_dir, "clean.image")
    fits_file = os.path.join(project_dir, "clean.fits")

    # Export to FITS
    exportfits(
        imagename=image_file,
        fitsimage=fits_file,
        overwrite=True
    )

    # 将FITS图像转换为NumPy数组并保存
    # 使用CASA的image analysis工具
    ia.open(image_file)
    image_data = ia.getchunk()
    ia.close()

    # 保存为NumPy文件
    np_file = os.path.join(numpy_dir, f"{project_name}_image.npy")
    np.save(np_file, image_data)

    # 保存元数据
    metadata_file = os.path.join(numpy_dir, f"{project_name}_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Project: {project_name}\n")
        f.write(f"Number of sources: {num_sources}\n")
        f.write(f"Central frequency: {central_freq}\n")
        f.write(f"Total bandwidth: {total_bandwidth} MHz\n")
        f.write(f"FITS image: {fits_file}\n")
        f.write(f"MS file: {ms_file}\n")

# 恢复原始工作目录
os.chdir(original_dir)

print(f"Simulation complete. Output saved to {base_dir}")
print(f"Data files are in: {data_dir}")
print(f"NumPy arrays are in: {numpy_dir}")