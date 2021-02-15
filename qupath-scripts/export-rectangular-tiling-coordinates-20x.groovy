createTiles = true

double frameWidth = 1392/5
double frameHeight = 1040/5
double overlap = 50
baseDirectory = PROJECT_BASE_DIR

clearDetections()
//Potentially store tiles as they are created
newTiles = []

//Store XY coordinates in an array

//Check all annotations. Use .findAll{expression} to select a subset
annotations = getAnnotationObjects()
imageName = getCurrentServer().getFile().getName()
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
            if(createTiles==true){createATile(x, y, frameWidth, frameHeight, overlap, roiA)}

            y = y+frameHeight-overlap
            xy << [x,y]
        }
        x = x+frameWidth-overlap
    }
    //Does not use CLASS of annotation in the name at the moment.
    path = buildFilePath(baseDirectory, "Tiles csv", imageName+i+".csv")
    new File(path).withWriter { fw ->
        fw.writeLine(header)
        //Make sure everything being sent is a child and part of the current annotation.
        
        xy.each{
            String line = it[0] as String +","+it[1] as String
            fw.writeLine(line)
        }
    }
}


//Comment out to avoid visual tiles.
if(createTiles==true){
    addObjects(newTiles)
    resolveHierarchy()
}
print " "
print "Output saved in  folder at " + tilePath


print "done"

def createATile(x,y,width,height, overlap, roiA) {
    def roi = new RectangleROI(x,y,width,height, ImagePlane.getDefaultPlane())
    if(roiA.getGeometry().intersects(roi.getGeometry())){
        newTiles << PathObjects.createDetectionObject(roi)
    }
}

import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI;