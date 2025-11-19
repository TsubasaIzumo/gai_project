import os
import glob
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime


def convert_tensorboard_to_txt(log_path, output_file=None):
    """
    Convert TensorBoard logs to txt format
    
    Args:
        log_path: TensorBoard log directory path or events file path
        output_file: Output txt file path, if None then auto-generated
    """
    # 判断是文件还是目录
    if os.path.isfile(log_path):
        # 如果是文件，直接使用该文件
        events_file = os.path.abspath(log_path)
        log_dir = os.path.dirname(events_file)
        # 如果 log_dir 为空（相对文件名），使用当前目录
        if not log_dir:
            log_dir = os.getcwd()
        print(f"Using events file: {events_file}")
    else:
        # 如果是目录，查找events文件
        log_dir = os.path.abspath(log_path)
        events_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not events_files:
            print(f"No events files found in {log_dir}")
            return None
        
        # Use the latest events file
        events_file = max(events_files, key=os.path.getmtime)
        print(f"Loading events file: {events_file}")
    
    # Load event file
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get all scalar tags
    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        print("No scalar data found")
        return None
    
    print(f"Found {len(scalar_tags)} metrics")
    
    # Generate output filename
    if output_file is None:
        if os.path.isfile(log_path):
            # 如果输入是文件，输出文件放在同一目录
            log_name = os.path.basename(os.path.dirname(log_dir))
        else:
            # 如果输入是目录
            log_name = os.path.basename(os.path.dirname(log_dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(log_dir, f"training_log_{log_name}_{timestamp}.txt")
    
    # Write to txt file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header information
        f.write("=" * 80 + "\n")
        f.write("Training Log - TensorBoard to Text Format\n")
        f.write("=" * 80 + "\n")
        f.write(f"Log Directory: {log_dir}\n")
        f.write(f"Events File: {os.path.basename(events_file)}\n")
        f.write(f"Conversion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Metrics: {len(scalar_tags)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write all metrics
        for tag in sorted(scalar_tags):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Metric: {tag}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"{'Step':<12} {'Value':<20} {'Wall Time':<20}\n")
            f.write("-" * 80 + "\n")
            
            try:
                events = ea.Scalars(tag)
                for event in events:
                    f.write(f"{event.step:<12} {event.value:<20.8f} {datetime.fromtimestamp(event.wall_time).strftime('%Y-%m-%d %H:%M:%S'):<20}\n")
                f.write(f"\nTotal Records: {len(events)}\n")
            except Exception as e:
                f.write(f"Read Error: {str(e)}\n")
        
        # Write summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("Log Conversion Complete\n")
        f.write("=" * 80 + "\n")
    
    print(f"Log saved to: {output_file}")
    return output_file


def read_training_log(log_path=None, output_file=None):
    """
    Read training log and convert to txt format
    
    Args:
        log_path: TensorBoard log directory or file path, if None then find the latest one
        output_file: Output file path
    """
    if log_path is None:
        # Find the latest log directory
        log_base = "lightning_logs"
        if not os.path.exists(log_base):
            print(f"Log directory {log_base} does not exist")
            return None
        
        # Find all version directories
        version_dirs = []
        for root, dirs, files in os.walk(log_base):
            if "version_0" in dirs:
                version_dirs.append(os.path.join(root, "version_0"))
        
        if not version_dirs:
            print("No log directories found")
            return None
        
        # Use the latest directory
        log_path = max(version_dirs, key=lambda x: os.path.getmtime(x))
        print(f"Using latest log directory: {log_path}")
    
    return convert_tensorboard_to_txt(log_path, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TensorBoard logs to txt format")
    parser.add_argument("log_path", type=str, nargs='?', default=None,
                       help="TensorBoard log directory or events file path (optional)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output txt file path (optional, auto-generated if not specified)")
    args = parser.parse_args()
    
    # If log_path is provided, use it; otherwise use the latest log
    if args.log_path:
        convert_tensorboard_to_txt(args.log_path, args.output)
    else:
        # Use the latest log directory
        read_training_log(output_file=args.output)