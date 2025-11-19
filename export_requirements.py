import subprocess

def export_requirements(output_file='requirements.txt'):
    try:
        with open(output_file, 'w') as f:
            subprocess.run(['pip', 'freeze'], stdout=f)
        print(f"✅ 成功导出依赖列表到 {output_file}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

if __name__ == '__main__':
    export_requirements()