createTiles = true

double pixelSizeSource = 1.105
double pixelSizeTarget = 0.222
double frameWidth = 1392 / pixelSizeSource * pixelSizeTarget
double frameHeight = 1040 / pixelSizeSource * pixelSizeTarget
double overlap = 50 * pixelSizeSource * pixelSizeTarget
baseDirectory = PROJECT_BASE_DIR

/***********************************************/

Logger logger = LoggerFactory.getLogger(QuPathGUI.class);

imageData = getQuPath().getImageData()
hierarchy = imageData.getHierarchy()
clearDetections()
//Potentially store tiles as they are created
newTiles = []

//Store XY coordinates in an array

//Check all annotations. Use .findAll{expression} to select a subset
annotations = hierarchy.getAnnotationObjects()
imageName = GeneralTools.getNameWithoutExtension(getQuPath().getProject().getEntry(imageData).getImageName())
logger.info(imageName.toString())
//Ensure the folder to store the csv exists
tilePath = buildFilePath(baseDirectory, "20x-tiles")
mkdirs(tilePath)
//CSV will be only two columns with the following header
String header="x_pos,y_pos";

annotations.eachWithIndex{a,i->
    xy = []
    roiA = a.getROI()
    //generate a bounding box to create tiles within
    bBoxX = a.getROI().getBoundsX()
    bBoxY = a.getROI().getBoundsY()
    bBoxH = a.getROI().getBoundsHeight()
    bBoxW = a.getROI().getBoundsWidth()
    x = bBoxX
   
    while (x< bBoxX+bBoxW){
        y = bBoxY
        while (y < bBoxY+bBoxH){
            if(createTiles==true){
                intersect=createATile(x, y, frameWidth, frameHeight, overlap, roiA)
            }
            if(intersect){xy << [x,y]}
            y = y+frameHeight-overlap

        }
        x = x+frameWidth-overlap
    }
    hierarchy.addPathObjects(newTiles)
    //Does not use CLASS of annotation in the name at the moment.
    annotationName = a.getName()
    path = buildFilePath(baseDirectory, "20x-tiles", imageName+"-"+annotationName+".csv")
    //logger.info(path.toString())
    new File(path).withWriter { fw ->
        fw.writeLine(header)
        //logger.info(header)
        //Make sure everything being sent is a child and part of the current annotation.
        
        xy.each{
            String line = it[0] as String +","+it[1] as String
            fw.writeLine(line)
        }
    }
}

boolean createATile(x,y,width,height, overlap, roiA) {
    def roi = new RectangleROI(x,y,width,height, ImagePlane.getDefaultPlane())
    if(roiA.getGeometry().intersects(roi.getGeometry())){
        newTiles << PathObjects.createDetectionObject(roi)
        return true;
    }else{ return false;}
}

import qupath.imagej.gui.IJExtension
import qupath.imagej.tools.IJTools
import qupath.lib.objects.PathObjectTools
import qupath.lib.regions.RegionRequest
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI;
import qupath.lib.gui.QuPathGUI

import static qupath.lib.gui.scripting.QPEx.*
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;