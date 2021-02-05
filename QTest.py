import os
import subprocess
script = "W:\\QuPath\\Scripts\\Export OMETIF from CLI.groovy"
image = "E:\\QuPath Active Projects\\QP Classifier Verification\\LuCa-7color1.tif"
qupath = "C:\\QuPath-0.2.2\\QuPath-0.2.2.exe"
subprocess.run([qupath, "script", script, "-i", image], shell = True)

print("What")
