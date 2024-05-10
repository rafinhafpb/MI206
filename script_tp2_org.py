import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, skeletonize
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

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
    else:
        return -1
    return operator


def ImAllOperations(img, se):
    imgs = []
    imDil = dilation(img, se)  # Dilatation morphologique
    imEro = erosion(img, se)  # Érosion morphologique
    imOuv = opening(img, se)  # Ouverture morphologique
    imFerm = closing(img, se)  # Fermeture morphologique

    gm = imDil - imEro  # Gradient morphologique
    imgs.append(gm)
    gminus = img - imEro  # Gradient intérieur
    imgs.append(gminus)
    gplus = imDil - img  # Gradient extérieur
    imgs.append(gplus)
    imTopHat1 = img - imOuv  # Top Hat
    imgs.append(imTopHat1)
    imTopHat2 = imFerm - img  # Top Hat conjugué
    imgs.append(imTopHat2)

    return imgs


def evaluate_operation(image, image_mask, structural_elements):
    best_f1 = 0
    for seuil in range(5,10):
        for se in structural_elements:
            images = ImAllOperations(image, se)
            count_operator = 0
            for img in images:
                img_seg = my_segmentation(img, image_mask, seuil)
                img_seg = np.logical_not(img_seg)
                img_seg[invalid_pixels] = 0

                ACCU, RECALL, img_skel, GT_skel = evaluate(img_seg.astype(np.int8), img_GT.astype(np.int8))
                #print(f"Seuil = {seuil}, Accuracy = {ACCU}, Recall = {RECALL}")
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

    return best_f1, best_accuracy, best_recall, best_operator, best_se, best_seuil, best_image, best_imageseg, best_skel, GT_skel


def evaluate_reconstruction(img):
    best_f1 = 0
    for i in range(1,5):
        for j in range(1,8,2):
            rec = reconstruction(erosion(img, square(i)), img, footprint=square(j))
            ACCU, RECALL, img_out_skel, _ = evaluate(rec.astype(np.uint8), img_GT.astype(np.uint8))
            f1_score = 2 * ACCU * RECALL / (ACCU + RECALL)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_si, best_sj = i, j
                best_image = rec
                best_skel = img_out_skel

    return best_f1, best_si, best_sj, best_image, best_skel


# Définition des éléments structurants
squares = [square(i) for i in range(4,8)]
diamands = [diamond(i) for i in range(4,8)]
se3 = np.random.randint(2, size=(6, 6))
se4 = disk(8)
se5 = np.array([[1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],], dtype=np.uint8)
se6 = np.array([[1, 0, 1]], dtype=np.bool_)
se7 = np.array([[1], [0], [1]], dtype=np.bool_)

structure_elements = [se3, se4, se5, se6, se7] + squares + diamands

img_number = [('01', 'OSC'), ('02', 'OSC'), ('03', 'OSN'), ('08', 'OSN'), ('21', 'OSC'), ('26', 'ODC'), ('28', 'ODN'), ('32', 'ODC'), ('37', 'ODN'), ('48', 'OSN')]

# Ouvrir l'image originale en niveau de gris
img_original = np.asarray(Image.open("./images_IOSTAR/star02_OSC.jpg")).astype(np.uint8)

nrows, ncols = img_original.shape
row, col = np.ogrid[:nrows, :ncols]

# On ne considere que les pixels dans le disque inscrit
img_mask = (np.ones(img_original.shape)).astype(np.bool_)
invalid_pixels = (row - nrows / 2) ** 2 + (col - ncols / 2) ** 2 > ((nrows - 10) / 2) ** 2
img_mask[invalid_pixels] = 0

# Ouvrir l'image Verite Terrain en booleen
img_GT = np.asarray(Image.open("./images_IOSTAR/GT_02.png")).astype(np.bool_)

# Operation de filtre moyenne sur l'image originale
img_filtered = filters.rank.mean(img_original, square(3))

# Évaluation de la meilleure operation, le meilleur élément structurant, le meilleur seuil et les images résultantes
f1_score, best_accuracy, best_recall, best_op, best_se, best_seuil, result_image, result_imag_seg, result_skel, GT_skel = evaluate_operation(img_filtered, img_mask, structure_elements)

print(f"\nBest f1: {f1_score}, Acc: {best_accuracy}, Rec: {best_recall}")
print(f"Best Treshold for Segmentation: {best_seuil}")
print(f"Best Structure Element: \n{best_se}")
print(f"Best Operation: {best_op}")

# Évaluation de la meilleure opération de reconstruction
f1_score_rec, best_si, best_sj, best_rec, result_skel_rec = evaluate_reconstruction(result_imag_seg)

print(f"\nReconstruction -> F1: {f1_score_rec}, square({best_si}), foot print = square({best_sj})")

plt.suptitle(best_op)
plt.subplot(231)
plt.imshow(img_original)
plt.title("Image Originale")
plt.subplot(232)
plt.imshow(result_imag_seg)
plt.title(f"Meilleure Résultat F-mesure: {f1_score*100:.2f}%")
plt.subplot(233)
plt.imshow(result_skel)
plt.title("Segmentation Squelette")
plt.subplot(234)
plt.imshow(result_image)
plt.title(f"Operation : {best_op}")
plt.subplot(235)
plt.imshow(img_GT)
plt.title("Verite Terrain")
plt.subplot(236)
plt.imshow(GT_skel)
plt.title("Verite Terrain Squelette")
plt.show()

plt.subplot(121)
plt.imshow(result_imag_seg)
plt.title(f"Sans Reconstruction F-masure:{f1_score*100:.2f}%")
plt.subplot(122)
plt.imshow(best_rec)
plt.title(f"Avec Reconstruction F-masure:{f1_score_rec*100:.2f}%")
plt.show()

# Résultats pour toutes les images
for number in img_number:
    img_original = np.asarray(Image.open(f"./images_IOSTAR/star{number[0]}_{number[1]}.jpg")).astype(np.uint8)
    img_GT = np.asarray(Image.open(f"./images_IOSTAR/GT_{number[0]}.png")).astype(np.bool_)
    # Operation de filtre moyenne sur l'image
    img_filtered = filters.rank.mean(img_original, square(3))

    # Évaluation de la meilleure operation, le meilleur élément structurant, le meilleur seuil et les images résultantes
    f1_score, best_accuracy, best_recall, best_op, best_se, best_seuil, result_image, result_imag_seg, result_skel, GT_skel = evaluate_operation(img_filtered, img_mask, structure_elements)
    print(f"\nImage: star{number[0]}_{number[1]}.jpg:")
    print(f"\nBest f1: {f1_score}, Acc: {best_accuracy}, Rec: {best_recall}")
    print(f"Best Treshold for Segmentation: {best_seuil}")
    print(f"Best Structure Element: \n{best_se}")
    print(f"Best Operation: {best_op}")

    # Évaluation de la meilleure opération de reconstruction
    f1_score_rec, best_si, best_sj, best_rec, result_skel_rec = evaluate_reconstruction(result_imag_seg)

    print(f"\nReconstruction -> F1: {f1_score_rec}, square({best_si}), foot print = square({best_sj})")
    plt.subplot(121)
    plt.imshow(best_rec)
    plt.title(f"Meilleure Résultat - F-mesure: {f1_score_rec*100:.2f}%")
    plt.subplot(122)
    plt.imshow(img_GT)
    plt.title("Verite Terrain")
    plt.suptitle(f"Comparaison Entre Résultat et Objectif - Image: star{number[0]}_{number[1]}.jpg")
    plt.show()