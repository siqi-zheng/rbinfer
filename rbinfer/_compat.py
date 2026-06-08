import os
import winreg

def find_rscript() -> str:
    """
    Find Rscript.exe on Windows by checking:
      1. System PATH  (works if R was added to PATH during install)
      2. Windows Registry  HKLM / HKCU  R-core keys
      3. Common default install locations as a last resort
    """
    import shutil

    # ── 1. Try PATH first ──────────────────────────────────────────────────────
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path

    # ── 2. Check Windows Registry for R install location ──────────────────────
    reg_keys = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\R-core\R"),
        (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\R-core\R"),
    ]
    for hive, key_path in reg_keys:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                rscript = os.path.join(install_path, "bin", "Rscript.exe")
                if os.path.isfile(rscript):
                    return rscript
        except (FileNotFoundError, OSError):
            continue

    # ── 3. Fallback: scan C:\Program Files\R\ for any installed version ────────
    base_dirs = [
        r"C:\Program Files\R",
        r"C:\Program Files (x86)\R",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\R"),
    ]
    candidates = []
    for base in base_dirs:
        if os.path.isdir(base):
            for version_dir in os.listdir(base):
                rscript = os.path.join(base, version_dir, "bin", "Rscript.exe")
                if os.path.isfile(rscript):
                    candidates.append(rscript)
    if candidates:
        # Pick the latest version (lexicographic sort is fine for R-4.x.y names)
        return sorted(candidates)[-1]

    raise FileNotFoundError(
        "Rscript.exe not found. Make sure R is installed, or add "
        r"'C:\Program Files\R\R-x.y.z\bin' to your system PATH."
    )