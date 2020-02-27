# ocr cut marks

Scripts for segmentation of marks on the iron plate. Using traditional OCR methods.

Including:


1.morphologyEx: to reinforce the contrast of the marks on the plate, meanwhile removing rust pattern on the plate.


2.Binarization： set pixels to 0 or 1 enforcing the mark on the plate.

OTSU: using otsu to select threshold

iteration: using value iteration to select threshold

CrossEntropy: calculate crossentropy to select threshold

3.vertical and horizontal projection: projecting the picture by gray value, cutting areas with marks on the plate. 

