//Script adjusted from https://forum.image.sc/t/importing-batch-annotations-xml-from-a-folder/37695/8?u=mike_nelson

path_to_aperio_xml = "F:/Documents/sp-project/smartpath_libraries/registered/"

clearAllObjects()
//Aperio Image Scope displays images in a different orientation
def rotated = false

def imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
def imageData = getCurrentImageData()
//Make sure the location you want to save the files to exists - requires a Project
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'image_export')
mkdirs(pathOutput)

def server = QP.getCurrentImageData().getServer()
def h = server.getHeight()
def w = server.getWidth()
print imageName
imageName = imageName.substring(0,imageName.indexOf("-"))
print imageName
//Prompt user for exported aperio image scope annotation file
//def path = path_to_aperio_xml+"Copy of "+imageName+ ".xml"
//path = path.substring(0, path.lastIndexOf(".")) + ".xml"  // HERE
path = path_to_aperio_xml+imageName.substring(imageName.lastIndexOf("_")+1) + ".xml"

def file = new File(path)
def text = file.getText()

def list = new XmlSlurper().parseText(text)
newAnnotations = []

list.Annotation.each {

    it.Regions.Region.each { region ->

        def tmp_points_list = []

        region.Vertices.Vertex.each{ vertex ->

            if (rotated) {
                X = vertex.@Y.toDouble()
                Y = h - vertex.@X.toDouble()
            }
            else {
                X = vertex.@X.toDouble()
                Y = vertex.@Y.toDouble()
            }
            tmp_points_list.add(new Point2(X, Y))
        }

        def roi = new PolygonROI(tmp_points_list)

        newAnnotations << PathObjects.createAnnotationObject(roi, getPathClass("AperioXML"))

    }
}

addObjects(newAnnotations)
/*getAnnotationObjects().eachWithIndex{anno,x->
    roi = anno.getROI()
    def requestROI = RegionRequest.createInstance(getCurrentServer().getPath(), 1, roi)
   
    pathOutput = buildFilePath(PROJECT_BASE_DIR, 'image_export', imageName+"_region_"+x)
    writeImageRegion(getCurrentServer(), requestROI, pathOutput+"_original.tif")
    writeImageRegion(getCurrentServer(), requestROI, pathOutput+"_original.png")
    writeImageRegion(getCurrentServer(), requestROI, pathOutput+"_original.jpg")
}*/

import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer
