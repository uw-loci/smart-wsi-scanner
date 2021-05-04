// baseDirectory = PROJECT_BASE_DIR

// Assuming you want full resolution, leave at 1
double downsample = 1.0 

server = getCurrentServer()
path= server.getBuilder().getURIs()[0].toString()
imageData = getCurrentImageData()
baseFilePath = path.substring(6, path.lastIndexOf(".")+1)
// Define output path (relative to project)
// def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())



pathOutput = buildFilePath(baseFilePath)
mkdirs(pathOutput)

new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(224)              // Define size of each tile, in pixels
    //.labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    //.annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    //.includePartialTiles(false)
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'
