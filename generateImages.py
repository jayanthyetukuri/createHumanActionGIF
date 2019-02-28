import os, cv2, math
from variables import sourceImagesDir, repository, action


videosDir = repository + action +'/dataSetVid/'
imagesDir = sourceImagesDir

def imagesFromVideos():
    cap = cv2.VideoCapture(videosDir)
    videoCount = 0
    for entry in os.listdir(videosDir):
        if entry.endswith('.avi'):
            videoCount += 1
            cap = cv2.VideoCapture(videosDir + entry)
            frameRate = cap.get(5)  # frame rate
            x = 1

            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % math.floor(frameRate) == 0):
                    filename = imagesDir + str(videoCount) + '_' + str(int(x)) + ".jpg"
                    x += 1
                    cv2.imwrite(filename, frame)

    cap.release()
    print ("Images from the videos generated...")
