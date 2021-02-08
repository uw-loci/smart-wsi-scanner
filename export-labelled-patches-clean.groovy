import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())


// Define output resolution
double requestedPixelSize = 10.0

// Convert to downsample
double downsample = 1.0 //requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
    .addLabel('Stroma', 2)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
annotations = getAnnotationObjects()
print annotations
removeObjects(annotations, true)

annotations.each{
    if(it.getPathClass() == null){return}
    addObject(it)

    className = it.getPathClass().toString()
    pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name, className)
    mkdirs(pathOutput)
    new TileExporter(imageData)
        .downsample(downsample)     // Define export resolution
        .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
        .tileSize(224)              // Define size of each tile, in pixels
        .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
        .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
        //.includePartialTiles(false)
        .overlap(0)                // Define overlap, in pixel units at the export resolution
        .writeTiles(pathOutput)     // Write tiles to the specified directory
    removeObject(it, true)
    print 'Done!'
}
addObjects(annotations)
fireHierarchyUpdate()
thing = annotations[0]
basePath = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)
baseDir = new File(basePath);
files = baseDir.listFiles();
classes = []
files.each{ classes<<it.getName().toString()}
classes.each{c->
    params = new ImageJMacroRunner(getQuPath()).getParameterList()
    tempPath= buildFilePath(PROJECT_BASE_DIR, 'tiles', name, c)
    dir = new File(tempPath);
    
    files = dir.listFiles().findAll{!it.getName().toString().contains("-labelled")};
    files.each{f->
        
        f= f.toString().replaceAll('\\\\', '/')
        //print f
        //print ParameterList.getParameterListJSON(params, ' ')
        macro = 'setBatchMode(true);n="'+f+'"; m=substring(n,0,n.length()-4); m=m+"-labelled.png"; open(m); getStatistics(nPixels, mean, min, max); if (max == 0){ close();File.delete(m);File.delete(n); };'

        ImageJMacroRunner.runMacro(params, imageData, null, thing, macro)
        
    }
    macro ='selectWindow("Log"); run("Close");'
    ImageJMacroRunner.runMacro(params, imageData, null, thing, macro)
}


import qupath.imagej.gui.ImageJMacroRunner
