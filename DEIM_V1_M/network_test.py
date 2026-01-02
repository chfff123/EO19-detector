import socket
import os
import subprocess

def main():
    print("=== Network and NCCL Test ===")
    print(f"Hostname: {socket.gethostname()}")
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        print(f"IP Address: {ip_address}")
    except:
        print("Could not resolve IP address")
    
    print("\n=== Port Check ===")
    port = 29500
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        print(f"Port {port} is available")
    except OSError as e:
        print(f"Port {port} in use: {str(e)}")
    finally:
        sock.close()
    
    print("\n=== NCCL Test with torchrun ===")
    # 创建测试脚本
    test_script = """
import torch
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="env://"
)

rank = dist.get_rank()
tensor = torch.ones(1).cuda() * (rank + 1)
dist.all_reduce(tensor)

print(f"Rank {rank}: NCCL communication successful. Result: {tensor.item()}")
"""
    
    script_path = "torchrun_nccl_test.py"
    with open(script_path, "w") as f:
        f.write(test_script)
    
    try:
        # 使用 torchrun 启动测试
        result = subprocess.run([
            "torchrun",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--standalone",
            script_path
        ], capture_output=True, text=True)
        
        print(f"\nTest exit code: {result.returncode}")
        print("Output:")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\nErrors:")
            print(result.stderr)
        
        print("="*50)
        print("Test Summary:")
        if result.returncode == 0:
            print("NCCL test: SUCCESS")
        else:
            print("NCCL test: FAILED")
    
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    main()