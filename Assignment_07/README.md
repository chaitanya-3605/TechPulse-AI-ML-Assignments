# Assignment 7: Pedestrian Detection using OpenCV (HOG + Linear SVM)

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Classical Computer Vision, Feature Descriptors & Object Detection

---

## 1. Project Overview & Objective
This assignment implements a classical pedestrian detection pipeline utilizing OpenCV's **Histogram of Oriented Gradients (HOG)** descriptor paired with a pre-trained **Linear Support Vector Machine (SVM)** people detector.

---

## 2. Technical Methodology
1. **Scene Ingestion:** Read input scene (`pedestrians_sample.jpg`) and convert color spaces for feature extraction.
2. **HOG Descriptor Setup:** Initialize `cv2.HOGDescriptor()` and attach pre-trained pedestrian SVM weights (`cv2.HOGDescriptor_getDefaultPeopleDetector()`).
3. **Multi-Scale Detection:** Execute `detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)` across image pyramid levels to detect pedestrians at various camera distances.
4. **Confidence Thresholding & Filtering:** Filter raw candidate windows with low SVM confidence scores (`weight < 0.50`) to suppress background false alarms.
5. **Non-Maximum Suppression (NMS):** Merge overlapping bounding boxes to yield clean pedestrian localization.

---

## 3. Empirical Results
- **Raw Detections:** 4 candidate bounding boxes identified, including 1 weak background candidate (`weight = 0.346`).
- **Post-Filter Detections:** 3 verified pedestrian bounding boxes (`weight >= 0.50`), matching all visible pedestrians in the scene with 0 false positives.

---

## 4. Execution Instructions
```bash
cd Assignments/Assignment_07
jupyter notebook TTL_Assignment_07.ipynb
```
