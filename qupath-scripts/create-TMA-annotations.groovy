imageData = getQuPath().getImageData()
hierarchy = imageData.getHierarchy()
cores = hierarchy.getTMAGrid().getTMACoreList()
coreAnno = []
cores.each {
    roi = it.getROI()
    coreName = it.getName().toString()
    temp = PathObjects.createAnnotationObject(roi,getPathClass("Tile"))
    temp.setName(coreName)
    coreAnno << temp
}
hierarchy.addPathObjects(coreAnno)