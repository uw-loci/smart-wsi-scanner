// source https://forum.image.sc/t/issue-with-bfconvert-and-big-jpg-images/43276/2

import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.images.servers.*
import javax.imageio.*

def path = getCurrentServer().getFile().toString()
print getCurrentServer().getPath()

def pathOutput = path.substring(0, path.lastIndexOf(".")+1) + "ome.tif"

println 'Reading image...'
def img = ij.IJ.openImage(path).getBufferedImage()
def server = new WrappedBufferedImageServer("Anything", img)

println 'Writing OME-TIFF'
new OMEPyramidWriter.Builder(server)
        .parallelize()
        .tileSize(512)
        .scaledDownsampling(1, 4)
        .build()
        .writePyramid(pathOutput)
        
println 'Done!'