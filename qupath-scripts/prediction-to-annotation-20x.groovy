//Also requires project for the file name, but the file path could be entered manually
//Using the same size tile I chose for the export
pixelSize20x = 0.222
pixelSize4x = 1.105
sizeH = 1040 * pixelSize20x / pixelSize4x
sizeW = 1392 * pixelSize20x / pixelSize4x

imageName = getProjectEntry().getImageName()
imageData = getCurrentImageData()
name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
newTiles = []
path = buildFilePath(PROJECT_BASE_DIR, "predictions", "20x-"+name+".csv")

new File(path).splitEachLine(",") {xy ->
    if(xy[0] == "x_pos"){return}
    
    //cycle through each line 
    double x = Double.parseDouble(xy[0])
    double y = Double.parseDouble(xy[1])

    def roi = new RectangleROI(x,y,sizeW,sizeH, ImagePlane.getDefaultPlane())
    
    newTiles << PathObjects.createDetectionObject(roi)
}
addObjects(newTiles)
resolveHierarchy()
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI;