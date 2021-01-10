def rois = getAnnotationObjects().collect {it.getROI()}
def gson = GsonTools.getInstance(true)
println gson.toJson(rois)

def name = getProjectEntry().getImageName() + '.json'
def path = buildFilePath(PROJECT_BASE_DIR, 'annotation results', name)
File file = new File(path)
file.write(gson.toJson(rois))