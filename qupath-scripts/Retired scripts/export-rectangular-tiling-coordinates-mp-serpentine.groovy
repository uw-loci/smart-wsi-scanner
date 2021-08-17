createTiles = true

double pixelSizeSource = 1.105
double pixelSizeTarget = 0.255
double frameWidth = 512 / pixelSizeSource * pixelSizeTarget
double frameHeight = 512 / pixelSizeSource * pixelSizeTarget
//Overlap percent - 10% is 10, not 0.1
double overlapPercent = 10
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
annotations = hierarchy.getAnnotationObjects().findAll{it.getPathClass()!=getPathClass("Background")}
imageName = GeneralTools.getNameWithoutExtension(getQuPath().getProject().getEntry(imageData).getImageName())
logger.info(imageName.toString())
//Ensure the folder to store the csv exists
tilePath = buildFilePath(baseDirectory, "mp-tiles")
mkdirs(tilePath)
//CSV will be only two columns with the following header
String header="x_pos,y_pos";

annotations.eachWithIndex{a,i->
    i= 0;
    index = 0;
    xy = [];
    yline = 0
    roiA = a.getROI()
    //generate a bounding box to create tiles within
    bBoxX = a.getROI().getBoundsX()
    bBoxY = a.getROI().getBoundsY()
    bBoxH = a.getROI().getBoundsHeight()
    bBoxW = a.getROI().getBoundsWidth()
    y = bBoxY
    x = bBoxX
    while (y< bBoxY+bBoxH){
        //In order to serpentine the resutls, there need to be two bounds for X now
        while ((x <= bBoxX+bBoxW) && (x >=bBoxX-frameWidth)){

            def roi = new RectangleROI(x,y,frameWidth,frameHeight, ImagePlane.getDefaultPlane())
            if(roiA.getGeometry().intersects(roi.getGeometry())){
                newAnno = PathObjects.createDetectionObject(roi)
                newAnno.setName(index.toString())
//                newAnno.getMeasurementList().putMeasurement(
                newTiles << newAnno
                xy << [x,y]
                print index + " good "+x
            }else {print x}
            if (yline%2 ==0){
                x = x+frameWidth-overlapPercent/100*frameWidth
            } else { x = x-(frameWidth - overlapPercent/100*frameWidth)}
            index++
        }
        y = y+frameHeight-overlapPercent/100*frameHeight
        if (yline%2 ==0){
            x = x-(frameWidth - overlapPercent/100*frameWidth)
         } else {x = x+frameWidth-overlapPercent/100*frameWidth}
        
        yline++
    }
    hierarchy.addPathObjects(newTiles)
    //Does not use CLASS of annotation in the name at the moment.
    annotationName = a.getName()
    path = buildFilePath(baseDirectory, "mp-tiles", imageName+"-"+annotationName+".csv")
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