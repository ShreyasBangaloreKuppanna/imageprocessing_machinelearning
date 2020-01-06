import os
import subprocess

# subprocess.call("pyside-uic.exe scratch_detector.ui -o scratch_detector_gui.py")
# subprocess.call("pyside-uic.exe about.ui -o about_gui.py")
# subprocess.call('pyside-rcc.exe -py3 icons.qrc -o icons_rc.py')


subprocess.call("pyinstaller --onefile --clean --name=\"Scratch Detector\" --icon=scratch.ico --windowed --uac-admin "
                "D:\shreyas_interview\\scratch_detector_main.py ")