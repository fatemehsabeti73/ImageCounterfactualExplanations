# -*- coding: utf-8 -*-
"""
Created on August 10, 2025

@author: FSM
"""
import os
import time
import numpy as np
import pandas as pd
import PIL.Image as Image
import cv2
from os import listdir
import tensorflow as tf
import tensorflow_hub as hub
from skimage.segmentation import quickshift

# --------- تنظیم مسیرها ----------
# فرض می‌کنیم این اسکریپت در فولدر کلون‌شده ریپو قرار دارد
# مسیر دیتاست تست
path_images_root = '/content/ImageCounterfactualExplanations/chihuahua_test'
classes = ['chihuahua_test']

# --------- import ماژول‌های داخلی ----------
from sedc_time import sedc_time
from explain_instance_lime import explain_instance_lime
from explain_instance_shap import explain_instance_shap
from perform_occlusion_analysis import perform_occlusion_analysis

# --------- بارگیری مدل ------------
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

# --------- دریافت لیبل‌های Imagenet ----------
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
imagenet_labels = np.array(open(labels_path).read().splitlines())

# --------- تابع لود عکس ----------
def loadImages(path, imshape):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img_path = os.path.join(path, image)
        img = Image.open(img_path).resize(imshape)
        img = np.array(img) / 255.0
        if img.shape == (224,224,3):
            loadedImages.append(img)
    return loadedImages

# --------- تابع محاسبه شباهت ----------
def calculate_similarity(segments_explanations):
    union = np.unique(segments_explanations)
    intersection = []
    for i in union:
        counter = 0
        for j in segments_explanations:
            if i in j:
                counter += 1
        if counter == len(segments_explanations):
            intersection.append(i)
    similarity = len(intersection) / len(union)
    return similarity

# --------- پیکربندی آزمایش ----------
table = pd.DataFrame(columns=[
    'Image', 'k_SEDC', 'similarity_sedc', 'similarity_lime', 'similarity_shap',
    'similarity_occlusion', 'mean_ct_sedc', 'mean_ct_lime', 'mean_ct_shap',
    'mean_ct_occlusion', 'times_counterfactual_lime', 'times_counterfactual_shap',
    'times_counterfactual_occlusion'
], index=[i for i in range(200)])

n = 0
n_runs = 2           # کم شده برای تست سریع
images_per_class = 5 # کم شده برای تست سریع
time_limit = 60

# --------- اجرای آزمایش ----------
for class_name in classes:
    path_images = path_images_root
    images = loadImages(path_images, IMAGE_SHAPE)
    print('Class ' + class_name + ' started')
    images = images[:images_per_class]  

    counter = 0
    for image in images:
        # پیش‌بینی
        result = classifier.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(result[0], axis=-1)

        # سگمنتیشن
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        perturbed_image = cv2.GaussianBlur(image, (31,31), 0)

        # --- SEDC ---
        segments_in_sedc_explanations = []
        computation_times_sedc = []
        too_long = False
        for i in range(n_runs):
            start = time.time()
            explanation, segments_in_explanation, _, _, too_long = sedc_time(
                image, classifier, segments, 'blur', time_limit
            )
            stop = time.time()
            if too_long:
                break
            n_segments = len(segments_in_explanation)
            segments_in_sedc_explanations.append(segments_in_explanation)
            computation_times_sedc.append(stop-start)
        if too_long:
            continue

        # --- LIME ---
        segments_in_lime_explanations = []
        computation_times_lime = []
        counter_lime = 0
        for i in range(n_runs):
            start = time.time()
            _, mask_lime = explain_instance_lime(image, classifier, n_segments)
            stop = time.time()
            segments_in_lime_explanations.append(np.unique(segments[mask_lime == 1]))
            computation_times_lime.append(stop-start)

            test_image = image.copy()
            for j in np.unique(segments[mask_lime == 1]):
                test_image[segments == j] = perturbed_image[segments == j]
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_lime += 1

        # --- SHAP ---
        segments_in_shap_explanations = []
        computation_times_shap = []
        counter_shap = 0
        for i in range(n_runs):
            start = time.time()
            _, segs_shap = explain_instance_shap(image, classifier, segments, n_segments)
            stop = time.time()
            segments_in_shap_explanations.append(segs_shap)
            computation_times_shap.append(stop-start)

            test_image = image.copy()
            for j in segs_shap:
                test_image[segments == j] = perturbed_image[segments == j]
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_shap += 1

        # --- Occlusion ---
        segments_in_occlusion_explanations = []
        computation_times_occlusion = []
        counter_occlusion = 0
        for i in range(n_runs):
            start = time.time()
            _, segs_occ = perform_occlusion_analysis(image, classifier, segments, n_segments)
            stop = time.time()
            segments_in_occlusion_explanations.append(segs_occ)
            computation_times_occlusion.append(stop-start)

            test_image = image.copy()
            for j in segs_occ:
                test_image[segments == j] = perturbed_image[segments == j]
            if np.argmax(classifier.predict(test_image[np.newaxis,...])) != predicted_class:
                counter_occlusion += 1

        # ذخیره در جدول
        table.loc[n] = [
            image, n_segments,
            calculate_similarity(segments_in_sedc_explanations),
            calculate_similarity(segments_in_lime_explanations),
            calculate_similarity(segments_in_shap_explanations),
            calculate_similarity(segments_in_occlusion_explanations),
            np.mean(computation_times_sedc),
            np.mean(computation_times_lime),
            np.mean(computation_times_shap),
            np.mean(computation_times_occlusion),
            counter_lime, counter_shap, counter_occlusion
        ]
        counter += 1
        n += 1
        print(f"Iteration {n} done")

# خروجی به اکسل
table.to_excel('/content/results_table.xlsx')
print("✅ Results saved to /content/results_table.xlsx")



