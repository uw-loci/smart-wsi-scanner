def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
println gson.toJson(annotations)

def name = getProjectEntry().getImageName() + '.json'
def path = buildFilePath(PROJECT_BASE_DIR, 'annotation results', name)
File file = new File(path)
file.write(gson.toJson(annotations))