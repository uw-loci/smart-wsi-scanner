//Name each annotation in the image by its XY centroids

getAnnotationObjects().each{

    it.setName((int)it.getROI().getCentroidX()+"_"+ (int)it.getROI().getCentroidY())

}