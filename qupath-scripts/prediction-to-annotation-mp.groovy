//Also requires project for the file name, but the file path could be entered manually
//Using the same size tile I chose for the export
pixelSizeShg = 0.509
pixelSize4x = 1.105
size = 256 * pixelSizeShg / pixelSize4x

imageName = getProjectEntry().getImageName()
imageData = getCurrentImageData()
name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
newTiles = []
path = buildFilePath(PROJECT_BASE_DIR, "predictions", "mp-"+name+".csv")

new File(path).splitEachLine(",") {xy ->
    if(xy[0] == "x_pos"){return}
    
    //cycle through each line 
    double x = Double.parseDouble(xy[0])
    double y = Double.parseDouble(xy[1])

    def roi = new RectangleROI(x,y,size,size, ImagePlane.getDefaultPlane())
    
    newTiles << PathObjects.createDetectionObject(roi)
}
addObjects(newTiles)
resolveHierarchy()
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI;