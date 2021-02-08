imageDirectory = "F:\\InputImages\\"
outputDirectory = 'F:\\ProcessedImages\\'
qupath = "F:\\Builds\\QuPath-0.2.3\\QuPath-0.2.3 (console).exe"

for file in fileList:
    if file.endswith('.tif'):
        imageFile = imageDirectory + file
        outputImage = outputDirectory + file
        subprocess.run([qupath, "convert-ome", imageFile, outputImage, "-y=4", "-p"], shell=True)
        os.rename(imageFile, outputImage)

