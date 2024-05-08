import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

se1 = square(6) # square

se2 = diamond(5) # diamond

#se3  = np.ones((3)) # line
se3 = np.random.randint(2, size=(6, 6))

se4 = disk(8)

se5 = np.array([[1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1]], dtype=np.uint8)

se6 = np.array([[1, 0, 1, 0, 1]], dtype=np.bool_)

se7 = np.array([[1], [0], [1]], dtype=np.bool_)

structure_elements = [se1, se2, se3, se4, se5, se6, se7]
print(type(structure_elements))

def my_segmentation(img, img_mask, seuil):
    img_out = (img_mask & (img < seuil))
    return img_out

def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

def switch(operator_counter):
    if operator_counter == 0:
        operator = 'Gradient'
    elif operator_counter == 1: 
        operator = 'Gradient Erosion'
    elif operator_counter == 2: 
        operator = 'Gradient Dilation'
    elif operator_counter == 3: 
        operator = 'Top Hat Ouverture'
    elif operator_counter == 4: 
        operator = 'Top Hat Fermeture'
    else:
        return -1
    return operator

#Ouvrir l'image originale en niveau de gris
img_original =  np.asarray(Image.open('C:/Users/rafin/Downloads/images_tp2/images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img_original.shape)

def ImAllOperations(img, se):
    imgs = []
    imDil = dilation(img, se) # Dilatation morphologique
    imEro = erosion(img, se) # Érosion morpholigique
    imOuv = dilation(imEro, se) # Ouverture morphologique
    imFerm = erosion(imDil, se) # Fermeture morphologique

    gm = imDil - imEro # Gradient morphologique
    imgs.append(gm)
    gminus = img - imEro # Gradient morphologique erosion
    imgs.append(gminus)
    gplus = imDil - img # Gadient morphologique dilation
    imgs.append(gplus)
    imTopHat1 = img - imOuv # Top Hat ouverture
    imgs.append(imTopHat1)
    imTopHat2 = imFerm # Top Hat fermeture
    imgs.append(imTopHat2)
    return imgs

imgs = ImAllOperations(img_original, se1)
gm, gminus, gplus, imTopHat1, imTopHat2 = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]

nrows, ncols = img_original.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img_original.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img_original,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('C:/Users/rafin/Downloads/images_tp2/images_IOSTAR/GT_01.png')).astype(np.bool_)

def evaluate_all(image, structural_elements, treshold):
    best_accuracy = 0
    for se in structural_elements:
        images = ImAllOperations(image, se)
        count_operator = 0
        for img in images:
            img[invalid_pixels] = 0
            img_seg = my_segmentation(img, img_mask, treshold)

            ACCU, RECALL, img_skel, GT_skel = evaluate(img_seg.astype(np.int8), img_GT.astype(np.int8))
            print('Accuracy =', ACCU,', Recall =', RECALL)
            #plt.imshow(img_seg)
            #plt.show()
            if ACCU > best_accuracy:
                best_accuracy = ACCU
                best_se = se
                best_image = img_seg
                best_operator = switch(count_operator)
            count_operator += 1
    return best_accuracy, best_se, best_operator, best_image
    
acuracy, melhorse, melhorop, melhorimg = evaluate_all(img_original, structure_elements, 75)
print(acuracy)
print(melhorse)
print(melhorop)
plt.imshow(melhorimg)
plt.show()

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out.astype(np.int8), img_GT.astype(np.int8))
print('Accuracy =', ACCU,', Recall =', RECALL)

'''
plt.subplot(231)
plt.imshow(img_original,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out)
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel)
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT)
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel)
plt.title('Verite Terrain Squelette')
plt.show()

plt.subplot(231)
plt.imshow(img_original,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(gm)
plt.title('Gradient Dilation - Erosion')
plt.subplot(233)
plt.imshow(gminus)
plt.title('Gadient Erosion')
plt.subplot(234)
plt.imshow(gplus)
plt.title('Gadient Dilation')
plt.subplot(235)
plt.imshow(imTopHat1)
plt.title('Top Hat Ourverture')
plt.subplot(236)
plt.imshow(imTopHat2)
plt.title('Top Hat Fermeture')
plt.show()
'''