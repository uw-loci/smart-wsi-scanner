double frameWidth = 100//1320
double frameHeight = 80//1040
double overlap = 10 //50
newTiles = []
annotations = getAnnotationObjects()
annotations.each{a->
    roiA = a.getROI()
    //generate a bounding box to create tiles within
    bBoxX = a.getROI().getBoundsX()
    bBoxY = a.getROI().getBoundsY()
    bBoxH = a.getROI().getBoundsHeight()
    bBoxW = a.getROI().getBoundsWidth()
    x = bBoxX
    
    print bBoxX
    print bBoxY
    
    while (x< bBoxX+bBoxW){
        y = bBoxY
        while (y < bBoxY+bBoxH){
            createATile(x, y, frameWidth, frameHeight, overlap, roiA)
            y = y+frameHeight-overlap
        }
        x = x+frameWidth-overlap
    }
    
    //Check each tile against the ROI of the annotation
    
    //add point to the list
    
}
addObjects(newTiles)

imageName = getProjectEntry().getImageName()
//Ensure the folder to store the csv exists
path = buildFilePath(PROJECT_BASE_DIR, "Tiles csv")
mkdirs(path)
String header="x_pos,y_pos";
//Cycle through each annotation and create a separate CSV per
getAnnotationObjects().eachWithIndex{it,i->
    //in the future the annotation name could be passed along with the list
    //pathClass = it.getPathClass()
    path = buildFilePath(PROJECT_BASE_DIR, "Tiles csv", imageName+" " +i+/*pathClass+*/".csv")
    new File(path).withWriter { fw ->
        fw.writeLine(header)
        //Make sure everything being sent is a child and part of the current annotation.
        tiles = it.getChildObjects().findAll{it.isDetection()}
        tiles.each{t->
            String line = t.getROI().getCentroidX() as String +","+t.getROI().getCentroidY()
            fw.writeLine(line)
        }
    }
}
print " "
print "Output saved in project folder at " + path


print "done"

def createATile(x,y,width,height, overlap, roiA) {
    def roi = new RectangleROI(x,y,width,height, ImagePlane.getDefaultPlane())
    if(roiA.getGeometry().intersects(roi.getGeometry())){
        newTiles << PathObjects.createDetectionObject(roi)
    }
}

import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI;