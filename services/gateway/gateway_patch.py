"""
gateway_patch.py
────────────────
Apply this patch to gateway.py to pass ?sid= to the STT service.
Without it, all sessions share one STT pipeline and corrupt each other.

Run:  python gateway_patch.py gateway.py
"""

import re, sys, shutil, pathlib

def patch(path: str):
    src = pathlib.Path(path).read_text(encoding="utf-8")  # ← fix

    # Check already patched
    if "STT_WS_URL}?sid=" in src or "stt_url" in src:
        print("✅ Already patched.")
        return

    old = (
        "                stt_ws = await _ws_connect(\n"
        "                    STT_WS_URL, max_retries=STT_MAX_RETRIES,"
    )
    new = (
        "                stt_url = f\"{STT_WS_URL}?sid={self.sid}\"\n"
        "                stt_ws = await _ws_connect(\n"
        "                    stt_url, max_retries=STT_MAX_RETRIES,"
    )

    if old not in src:
        print("❌ Could not find target line — patch failed.")
        print("   Apply manually: change STT_WS_URL → f\"{STT_WS_URL}?sid={self.sid}\"")
        sys.exit(1)

    patched = src.replace(old, new, 1)
    shutil.copy(path, path + ".bak")
    pathlib.Path(path).write_text(patched, encoding="utf-8")  # ← fix
    print(f"✅ Patched {path}  (backup → {path}.bak)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "gateway.py"
    patch(target)