import os
import sys
import subprocess
import time
import psutil
import torch

# ======================================================
# 🔐 CRITICAL GPU PROCESSES (DO NOT TOUCH)
# ======================================================

CRITICAL_KEYWORDS = {
    "nvidia",
    "nvdisplay",
    "nvcontainer",
    "system",
    "wininit",
    "winlogon",
    "dwm",
    "csrss",
    "services",
    "lsass",
    "smss",
    "svchost",
    "explorer",
}

# ======================================================
# 🧠 HELPERS
# ======================================================

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
    except:
        return ""

def is_critical(name: str, pid: int) -> bool:
    """Check if process is critical"""
    name = name.lower()
    
    # Never touch current Python process
    if pid == os.getpid():
        return True
    
    # Check against critical keywords
    return any(k in name for k in CRITICAL_KEYWORDS)

def get_gpu_processes():
    out = run(
        "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory "
        "--format=csv,noheader,nounits"
    )

    procs = []
    for line in out.splitlines():
        try:
            parts = line.split(",")
            if len(parts) >= 3:
                pid = int(parts[0].strip())
                name = parts[1].strip()
                mem = int(parts[2].strip())
                procs.append({
                    "pid": pid,
                    "name": name,
                    "mem": mem
                })
        except Exception as e:
            print(f"⚠️ Warning parsing line: {line} - {e}")
    return procs

# ======================================================
# 🔄 SAFE CPU-RESTART STRATEGY
# ======================================================

def restart_cpu_only(proc):
    """
    Safely restart a process in CPU mode.
    Returns: (success, reason)
    """
    pid = proc["pid"]
    name = proc["name"]
    
    print(f"\n🔄 Processing: {name} ({proc['mem']} MB)")

    try:
        p = psutil.Process(pid)
        
        # Get process info BEFORE killing
        try:
            exe = p.exe()
            cmdline = p.cmdline()
            cwd = p.cwd()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            print(f"   ❌ Cannot access process info (need admin rights)")
            return False, "insufficient_permissions"
        
        print(f"   📋 Path: {exe}")
        
        # Terminate process
        print(f"   🛑 Stopping...")
        p.terminate()
        
        try:
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            print(f"   ⚠️  Force killing...")
            p.kill()
            p.wait(timeout=2)
        
    except psutil.NoSuchProcess:
        print(f"   ℹ️  Process already ended")
        return True, "already_ended"
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e)

    time.sleep(1)

    # Try to restart in CPU mode
    if exe and os.path.exists(exe):
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            
            # Determine CPU flags based on process type
            cpu_flags = []
            if "chrome" in name.lower() or "edge" in name.lower():
                cpu_flags = ["--disable-gpu", "--disable-software-rasterizer"]
            
            print(f"   🔄 Restarting in CPU mode...")
            subprocess.Popen(
                [exe] + cpu_flags,
                env=env,
                cwd=cwd if cwd else None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print(f"   ✅ Restarted in CPU mode")
            return True, "restarted"
            
        except Exception as e:
            print(f"   ❌ Could not restart: {e}")
            return False, f"restart_failed: {e}"
    else:
        print(f"   ⚠️  Process killed (executable not found)")
        return False, "exe_not_found"

# ======================================================
# 🧹 SAFE GPU CLEANER WITH WAITING
# ======================================================

def free_gpu_smart(min_mem_mb=100):
    """
    Safely free GPU memory.
    WAITS for user to close apps manually, then proceeds.
    """
    print("\n🧹 SAFE GPU MEMORY RECLAIMER")
    print("=" * 60)

    # Get GPU processes
    gpu_procs = get_gpu_processes()

    if not gpu_procs:
        print("✅ No GPU processes found - GPU is clean")
        return 0

    # Categorize processes
    critical_procs = []
    small_procs = []
    movable_procs = []
    
    for proc in gpu_procs:
        if is_critical(proc["name"], proc["pid"]):
            critical_procs.append(proc)
        elif proc["mem"] < min_mem_mb:
            small_procs.append(proc)
        else:
            movable_procs.append(proc)
    
    # Report findings
    print(f"\n📊 Found {len(gpu_procs)} GPU processes:")
    print(f"   🔒 Critical (protected): {len(critical_procs)}")
    print(f"   🟡 Small usage (<{min_mem_mb}MB): {len(small_procs)}")
    print(f"   🎯 Can be moved: {len(movable_procs)}")
    
    if critical_procs:
        print(f"\n🔒 PROTECTED processes (won't touch):")
        for proc in critical_procs:
            print(f"   • {proc['name']} ({proc['mem']}MB)")
    
    if small_procs:
        print(f"\n🟡 SMALL usage (ignoring):")
        for proc in small_procs:
            print(f"   • {proc['name']} ({proc['mem']}MB)")
    
    if not movable_procs:
        print("\n✅ No processes to move")
        return 0
    
    # ===================================================
    # 🎯 SHOW WHAT NEEDS TO BE CLOSED (IN WHITE/DEFAULT)
    # ===================================================
    print(f"\n" + "=" * 60)
    print("🎯 PROCESSES USING GPU MEMORY:")
    print("=" * 60)
    
    total_potential = 0
    for i, proc in enumerate(movable_procs, 1):
        # Print in default color (white/system default)
        print(f"{i}. {proc['name']}")
        print(f"   Memory: {proc['mem']}MB")
        print(f"   PID: {proc['pid']}")
        print()
        total_potential += proc['mem']
    
    print(f"💡 Total memory that can be freed: ~{total_potential}MB")
    print("=" * 60)
    
    # ===================================================
    # ⏸️ WAIT FOR USER ACTION
    # ===================================================
    print("\n⏸️  OPTIONS:")
    print("   1. Close these apps manually and press ENTER")
    print("   2. Type 'auto' to let script restart them automatically")
    print("   3. Type 'skip' to continue without freeing memory")
    
    choice = input("\n👉 Your choice: ").strip().lower()
    
    if choice == 'skip':
        print("⏭️  Skipped GPU cleanup")
        return 0
    
    elif choice == 'auto':
        # Automatic mode - restart processes
        print("\n🤖 AUTOMATIC MODE - Restarting processes...")
        print("⚠️  This will close and restart these applications")
        print("   You may lose unsaved work!")
        
        confirm = input("\n   Confirm? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("❌ Cancelled")
            return 0
        
        print(f"\n🔄 Moving {len(movable_procs)} processes to CPU...")
        success_count = 0
        
        for proc in movable_procs:
            success, reason = restart_cpu_only(proc)
            if success:
                success_count += 1
        
        print(f"\n📊 Results: {success_count}/{len(movable_procs)} processes moved")
        
    else:
        # Manual mode - wait for user to close apps
        print("\n⏳ WAITING for you to close the apps listed above...")
        print("   Press ENTER when done")
        input()
        
        print("\n🔍 Checking if apps were closed...")
        
    # ===================================================
    # 🔍 VERIFY GPU IS CLEAN
    # ===================================================
    time.sleep(2)
    
    new_gpu_procs = get_gpu_processes()
    new_movable = [p for p in new_gpu_procs 
                   if not is_critical(p["name"], p["pid"]) 
                   and p["mem"] >= min_mem_mb]
    
    if not new_movable:
        print("✅ GPU memory successfully freed!")
        return len(movable_procs)
    else:
        print(f"\n⚠️  Still {len(new_movable)} processes using GPU:")
        for proc in new_movable:
            print(f"   • {proc['name']} ({proc['mem']}MB)")
        return len(movable_procs) - len(new_movable)

# ======================================================
# 🔥 FORCE GPU (NO FALLBACK)
# ======================================================

def force_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not torch.cuda.is_available():
        print("❌ CUDA NOT AVAILABLE — EXITING")
        sys.exit(1)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    print("\n🚀 GPU FORCED (cuda:0)")
    print(f"🧠 GPU: {torch.cuda.get_device_name(0)}")
    
    free_mem = torch.cuda.mem_get_info()[0] // 1024**2
    total_mem = torch.cuda.mem_get_info()[1] // 1024**2
    
    print(f"📦 GPU Memory:")
    print(f"   Free: {free_mem}MB")
    print(f"   Total: {total_mem}MB")
    print(f"   Used: {total_mem - free_mem}MB")
    
    return torch.device("cuda:0")


def get_gpu_memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return None
    
    free_mem = torch.cuda.mem_get_info()[0] // 1024**2
    total_mem = torch.cuda.mem_get_info()[1] // 1024**2
    used_mem = total_mem - free_mem
    
    return {
        'free_mb': free_mem,
        'total_mb': total_mem,
        'used_mb': used_mem,
        'utilization': (used_mem / total_mem * 100) if total_mem > 0 else 0
    }


def cleanup_gpu_memory():
    """Cleanup GPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ======================================================
# 🚀 SAFE EXECUTION WITH WAIT
# ======================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SAFE GPU MEMORY OPTIMIZER")
    print("   (Wait-and-Process Mode)")
    print("=" * 60)
    
    # Check admin rights
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if is_admin:
            print("✅ Running as Administrator")
        else:
            print("⚠️  Not Administrator - some features may be limited")
    except:
        pass
    
    print("=" * 60)
    
    # Step 1: Show what's using GPU and WAIT
    freed_count = free_gpu_smart(min_mem_mb=100)
    
    # Step 2: Force GPU and show available memory
    print("\n" + "=" * 60)
    force_gpu()
    
    # Step 3: Ready for your workload
    print("\n" + "=" * 60)
    print("🤖 GPU IS READY FOR YOUR WORKLOAD")
    print("=" * 60)
    
    # --------------------------------------------------
    # 🔥 YOUR GPU MODEL / WORKLOAD STARTS HERE
    # --------------------------------------------------
    print("\n💡 You can now run your GPU code...")
    print("   Example:")
    print("   model = YourModel().cuda()")
    print("   output = model(input)")