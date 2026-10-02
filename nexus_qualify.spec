# PyInstaller spec for the future one-binary seller executable.
# Built per-OS on CI (a macOS host cannot emit a Windows .exe).
#   pyinstaller nexus_qualify.spec
# The .exe embeds the pinned runtime; official assets ship alongside it and are
# verified by hash at startup exactly like the ZIP distribution.
# Status: EXPERIMENTAL — the supported seller path today is the dist ZIP +
# pinned-requirements launcher (fully offline after one-time setup).
block_cipher = None

a = Analysis(
    ["run_qualify.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("profiles", "profiles"),
        ("assets", "assets"),
        ("README_FIRST.txt", "."),
        ("requirements.lock", "."),
    ],
    hiddenimports=["nexus_bench.cli", "nexus_bench.qualify", "ultralytics"],
    excludes=["pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="NEXUS_Qualification",
    console=True,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="NEXUS_Qualification")
