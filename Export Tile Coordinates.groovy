//Creates a CSV file per annotation object that contains a list of tile centroids.
//Centroids are in pixels from the upper left hand corner of the image.

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
        tiles = it.getChildObjects().findAll{it.isTile()}
        tiles.each{t->
            String line = t.getROI().getCentroidX() as String +","+t.getROI().getCentroidY()
            fw.writeLine(line)
        }
    }
}
print " "
print "Output saved in project folder at " + path
