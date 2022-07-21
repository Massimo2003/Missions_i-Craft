from DeepImageSearch import Index, LoadData, SearchImage

def loadData(path):
    return LoadData().from_folder([path])

ImagesData = loadData("AllImages")
ImagesData[:20]

def indexImages(data):
    Index(data).Start()
    
# indexImages(ImagesData)

def similarImage(imgToComp, nbImages):
    SearchImage().get_similar_images(image_path = imgToComp, number_of_images = nbImages)
    
similarImage(ImagesData[2], 3)

def plotSimilarImages(imgToComp):
    SearchImage().plot_similar_images(image_path = imgToComp)
    
plotSimilarImages(ImagesData[2])