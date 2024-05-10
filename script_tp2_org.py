import numpy as np
from skimage.morphology import (
    erosion,
    dilation,
    binary_erosion,
    opening,
    closing,
    white_tophat,
    reconstruction,
    black_tophat,
    skeletonize,
    convex_hull_image,
    thin,
)
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

se1 = square(6)  # square
se2 = diamond(5)  # diamond
se3 = np.random.randint(2, size=(6, 6))
se4 = disk(8)
se5 = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)
se6 = np.array([[1, 0, 1, 0, 1]], dtype=np.bool_)
se7 = np.array([[1], [0], [1]], dtype=np.bool_)

structure_elements = [se1, se2, se3, se4, se5, se6, se7]


def my_segmentation(img, img_mask, seuil):
    img_out = img_mask & (img < seuil)
    return img_out


def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT)  # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out)  # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT)  # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT)  # Faux positifs
    FN = np.sum(GT_skel & ~img_out)  # Faux negatifs

    ACCU = TP / (TP + FP)  # Precision
    RECALL = TP / (TP + FN)  # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel


def switch(operator_counter):
    if operator_counter == 0:
        operator = "Gradient"
    elif operator_counter == 1:
        operator = "Gradient Erosion"
    elif operator_counter == 2:
        operator = "Gradient Dilation"
    elif operator_counter == 3:
        operator = "Top Hat Ouverture"
    elif operator_counter == 4:
        operator = "Top Hat Fermeture"
    elif operator_counter == 5:
        operator = "Ouverture after Gradient"
    elif operator_counter == 6:
        operator = "Fermeture after Gradient"
    elif operator_counter == 7:
        operator = "Ouverture after Gradient Erosion"
    elif operator_counter == 8:
        operator = "Fermeture after Gradient Erosion"
    elif operator_counter == 9:
        operator = "Ouverture after Gradient Dilation"
    elif operator_counter == 10:
        operator = "Fermeture after Gradient Dilation"
    else:
        return -1
    return operator


def ImAllOperations(img, se):
    imgs = []
    imDil = dilation(img, se)  # Dilatation morphologique
    imEro = erosion(img, se)  # Érosion morpholigique
    imOuv = opening(img, se)  # Ouverture morphologique
    imFerm = closing(img, se)  # Fermeture morphologique

    gm = imDil - imEro  # Gradient morphologique
    imgs.append(gm)
    gminus = img - imEro  # Gradient morphologique erosion
    imgs.append(gminus)
    gplus = imDil - img  # Gradient morphologique dilation
    imgs.append(gplus)
    imTopHat1 = img - imOuv  # Top Hat ouverture
    imgs.append(imTopHat1)
    imTopHat2 = imFerm - img  # Top Hat fermeture
    imgs.append(imTopHat2)
    operationTest1 = opening(gm)  # Ouverture après Gradient
    imgs.append(operationTest1)
    operationTest2 = closing(gm)  # Fermeture après Gradient
    imgs.append(operationTest2)
    operationTest3 = closing(gminus)  # Fermeture après Gradient morphologique erosion
    imgs.append(operationTest3)
    operationTest4 = opening(gminus)  # Ouverture après Gradient morphologique erosion
    imgs.append(operationTest4)
    operationTest5 = closing(gplus)  # Fermeture après Gradient morphologique dilation
    imgs.append(operationTest5)
    operationTest6 = opening(gplus)  # Ouverture après Gradient morphologique dilation
    imgs.append(operationTest6)

    return imgs


def evaluate_all(image, structural_elements):
    best_f1 = 0
    best_accuracy = 0
    best_recall = 0
    for seuil in [14]:
        for se in structural_elements:
            images = ImAllOperations(image, se)
            count_operator = 0
            for img in images:
                if count_operator != 11:
                    img[invalid_pixels] = 0
                    img_seg = my_segmentation(img, img_mask, seuil)
                    img_seg = np.logical_not(img_seg)
                    img_seg[invalid_pixels] = 0
                else:
                    img_seg = img

                ACCU, RECALL, img_skel, GT_skel = evaluate(
                    img_seg.astype(np.int8), img_GT.astype(np.int8)
                )

                print(seuil, "Accuracy =", ACCU, ", Recall =", RECALL)

                f1score = 2*ACCU * RECALL / (ACCU + RECALL)

                if f1score > best_f1:
                    best_accuracy = ACCU
                    best_recall = RECALL
                    best_f1 = f1score
                    best_se = se
                    best_image = img
                    best_imageseg = img_seg
                    best_operator = switch(count_operator)
                    best_seuil = seuil
                    best_skel = img_skel

                count_operator += 1
    return (
        best_f1,
        best_accuracy,
        best_recall,
        best_operator,
        best_se,
        best_seuil,
        best_image,
        best_imageseg,
        best_skel,
        GT_skel,
    )


# Ouvrir l'image originale en niveau de gris
img_original = np.asarray(Image.open("./TP2/images_IOSTAR/star08_OSN.jpg")).astype(
    np.uint8
)

nrows, ncols = img_original.shape
row, col = np.ogrid[:nrows, :ncols]

# On ne considere que les pixels dans le disque inscrit
img_mask = (np.ones(img_original.shape)).astype(np.bool_)
invalid_pixels = (row - nrows / 2) ** 2 + (col - ncols / 2) ** 2 > (
    (nrows - 20) / 2
) ** 2
img_mask[invalid_pixels] = 0

# Ouvrir l'image Verite Terrain en booleen
img_GT = np.asarray(Image.open("./TP2/images_IOSTAR/GT_08.png")).astype(np.bool_)

(
    f1_score,
    best_accuracy,
    best_recall,
    melhorop,
    melhorse,
    best_seuil,
    melhorimg,
    melhorimgseg,
    best_skel,
    GT_skel,
) = evaluate_all(img_original, structure_elements)

print(f"Best f1: {f1_score}, Acc: {best_accuracy}, Rec: {best_recall}")
print(best_seuil)
print(melhorse)
print(melhorop)

rec = reconstruction(erosion(melhorimgseg, square(3)), melhorimgseg, footprint=square(3))

ACCU, RECALL, img_out_skel, GT_skel = evaluate(rec.astype(np.uint8), img_GT.astype(np.uint8))
f1_score = 2*ACCU * RECALL / (ACCU + RECALL)

print(f"Reconstruction -> F1: {f1_score}, Acc: {ACCU}, Rec: {RECALL}")


plt.suptitle(melhorop)
plt.subplot(231)
plt.imshow(img_original, cmap="gray")
plt.title("Image Originale")
plt.subplot(232)
plt.imshow(melhorimg)
plt.title("Segmentation")
plt.subplot(233)
plt.imshow(melhorimgseg)
plt.title("Segmentation binary")
plt.subplot(234)
plt.imshow(best_skel)
plt.title("Segmentation squelette")
plt.subplot(235)
plt.imshow(img_GT)
plt.title("Verite Terrain")
plt.subplot(236)
plt.imshow(GT_skel)
plt.title("Verite Terrain Squelette")
plt.show()

plt.subplot(121)
plt.imshow(melhorimgseg)
plt.title("Antes")
plt.subplot(122)
plt.imshow(rec)
plt.title("Reconstruction")
plt.show()
