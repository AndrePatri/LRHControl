import pynvml
import time

# Initialize NVML
pynvml.nvmlInit()

# Get the first GPU handle
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

def get_gpu_utilization():
    # Get memory statistics
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Get PCIe throughput (bytes/second)
    rx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
    tx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)

    return memory_info, rx_throughput, tx_throughput

try:
    while True:
        memory_info, rx_throughput, tx_throughput = get_gpu_utilization()
        print(f"GPU Memory Used: {memory_info.used / (1024**2):.2f} MB")
        print(f"PCIe RX (CPU to GPU): {rx_throughput / 1024:.2f} KB/s")
        print(f"PCIe TX (GPU to CPU): {tx_throughput / 1024:.2f} KB/s")
        time.sleep(1)  # Sample every second

finally:
    pynvml.nvmlShutdown()
