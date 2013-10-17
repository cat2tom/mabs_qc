#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
This Python script shows inside a web browser the results of the image registrations executed by the Multi Altas Based Segmentation algorithm embedded in Plastimatch.
In this way the user can quickly check if some registration failed.

Author: Paolo Zaffino (p DOT zaffino AT unicz DOT it)
NOT TESTED ON PYTHON 3

For help: ./mabs_qc.py --help
"""

from __future__ import division
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
from scipy import ndimage
import shutil
import SimpleITK as sitk
import webbrowser

## Parser settings
parser = argparse.ArgumentParser(description='Registration quality check tool for Plastimatch implementation of Multi Atlas Based Segmentation')
parser.add_argument('--mabs_cfg', help='Mabs configuration file', type=str, required=True)
parser.add_argument('--output_reports_dir', help='Output reports root directory', type=str, required=True)
parser.add_argument('--browser', help='Web browser used to show the images, default is firefox', type=str, default="firefox")
parser.add_argument('--alpha', help='Image trasparency (default is 0.7)', type=float, default=0.7)
args = parser.parse_args()

def _compute_rotation(direction_cosines, index_on_diag, other_index):
    if direction_cosines[other_index] == 0:
        return 0
    elif direction_cosines[other_index] != 0:
        return np.degrees(np.arctan(float(direction_cosines[index_on_diag])/float(direction_cosines[other_index]))
                          * (float(direction_cosines[index_on_diag])/float(np.abs(direction_cosines[index_on_diag]))))

def _rotate_and_mask(img, angle):
    img=ndimage.interpolation.rotate(img, angle, order=0)
    img=np.ma.masked_where(img <= (np.amin(img) + np.abs(np.amin(img)*0.05)), img)
    return img

def create_screenshot (ref_img_fn, warp_img_fn, report_fn):
    
    subject_name = ref_img_fn.split(os.sep)[-2]
    deformed_atlas_name =  warp_img_fn.split(os.sep)[-3]

    print("Creating report for subject %s, atlas %s" % (subject_name, deformed_atlas_name))
    
    reference_img = sitk.ReadImage(ref_img_fn)
    warped_img = sitk.ReadImage(warp_img_fn)

    # Compute middle slices indexes
    middle_axial_slice_number=int(reference_img.GetSize()[2]/2.0)
    middle_coronal_slice_number=int(reference_img.GetSize()[1]/2.0)
    middle_sagittal_slice_number=int(reference_img.GetSize()[0]/2.0)

    # Aspect ratio for plot
    axial_ratio=reference_img.GetSpacing()[1]/float(reference_img.GetSpacing()[0])
    coronal_ratio=reference_img.GetSpacing()[2]/float(reference_img.GetSpacing()[1])
    sagittal_ratio=reference_img.GetSpacing()[2]/float(reference_img.GetSpacing()[0])

    # Axial rotations
    reference_axial_rotation = _compute_rotation(reference_img.GetDirection(), 0, 1)
    warped_axial_rotation = _compute_rotation(warped_img.GetDirection(), 0, 1)
    # Coronal rotation
    reference_coronal_rotation = _compute_rotation(reference_img.GetDirection(), 4, 5)
    warped_coronal_rotation = _compute_rotation(warped_img.GetDirection(), 4, 5)
    # Sagittal rotation
    reference_sagittal_rotation = _compute_rotation(reference_img.GetDirection(), 8, 6)
    warped_sagittal_rotation = _compute_rotation(warped_img.GetDirection(), 8, 6)

    # Set colormaps
    cmap_subject = mpl.cm.winter
    cmap_atlas = mpl.cm.autumn
    
    # Axial subplot
    plt.subplot(131)
    reference_middle_axial_slice=sitk.GetArrayFromImage(reference_img).T[:,::-1,middle_axial_slice_number].T # Extract reference slice
    warped_middle_axial_slice=sitk.GetArrayFromImage(warped_img).T[:,::-1,middle_axial_slice_number].T # Extract warped slice
    axial_background=np.zeros_like(reference_middle_axial_slice) # Create black background
    reference_middle_axial_slice=_rotate_and_mask(reference_middle_axial_slice, reference_axial_rotation) # Rotate and mask reference image
    warped_middle_axial_slice=_rotate_and_mask(warped_middle_axial_slice, warped_axial_rotation) # Rotate and mask warped image
    plt.imshow(ndimage.interpolation.rotate(axial_background, reference_axial_rotation, order=0), aspect=coronal_ratio, cmap=mpl.cm.gray) # Show black rotated background
    plt.imshow(reference_middle_axial_slice, aspect=axial_ratio, cmap=cmap_subject) # Show reference image
    plt.imshow(warped_middle_axial_slice, aspect=axial_ratio, alpha=args.alpha, cmap=cmap_atlas) # Show warped image
    plt.axis('off')

    # Coronal subplot
    plt.subplot(132, title = "Subject = %s\nDeformed atlas = %s" % (subject_name, deformed_atlas_name))
    reference_middle_coronal_slice=sitk.GetArrayFromImage(reference_img).T[:,middle_coronal_slice_number,::-1].T # Extract reference slice
    warped_middle_coronal_slice=sitk.GetArrayFromImage(warped_img).T[:,middle_coronal_slice_number,::-1].T # Extract warped slice
    coronal_background=np.zeros_like(reference_middle_coronal_slice) # Create black background
    reference_middle_coronal_slice=_rotate_and_mask(reference_middle_coronal_slice, reference_coronal_rotation) # Rotate and mask reference image
    warped_middle_coronal_slice=_rotate_and_mask(warped_middle_coronal_slice, warped_coronal_rotation) # Rotate and mask warped image
    plt.imshow(ndimage.interpolation.rotate(coronal_background, reference_coronal_rotation, order=0), aspect=coronal_ratio, cmap=mpl.cm.gray) # Show black rotated background
    plt.imshow(reference_middle_coronal_slice, aspect=coronal_ratio, cmap=cmap_subject) # Show reference image
    plt.imshow(warped_middle_coronal_slice, aspect=coronal_ratio, alpha=args.alpha, cmap=cmap_atlas) # Show warped image
    plt.axis('off')
    
    # Sagittal subplot
    plt.subplot(133)
    reference_middle_sagittal_slice=sitk.GetArrayFromImage(reference_img).T[middle_sagittal_slice_number,:,::-1].T # Extract reference slice
    warped_middle_sagittal_slice=sitk.GetArrayFromImage(warped_img).T[middle_sagittal_slice_number,:,::-1].T # Extract warped slice
    sagittal_background=np.zeros_like(reference_middle_sagittal_slice) # Create black background
    reference_middle_sagittal_slice=_rotate_and_mask(reference_middle_sagittal_slice,reference_sagittal_rotation) # Rotate and mask reference image
    warped_middle_sagittal_slice=_rotate_and_mask(warped_middle_sagittal_slice, warped_sagittal_rotation) # Rotate and mask warped image
    plt.imshow(ndimage.interpolation.rotate(sagittal_background, reference_sagittal_rotation), aspect=sagittal_ratio, cmap=mpl.cm.gray) # Show black rotated background
    plt.imshow(reference_middle_sagittal_slice, aspect=sagittal_ratio, cmap=cmap_subject) # Show reference image
    plt.imshow(warped_middle_sagittal_slice, aspect=sagittal_ratio, alpha=args.alpha, cmap=cmap_atlas) # Show warped image
    plt.axis('off')

    # Save image
    fig = plt.gcf()
    fig.set_size_inches(25,20)
    plt.savefig(report_fn, bbox_inches="tight")
    fig.clf()

def list_only_dirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# Parse mabs cfg file
cfg_file = open (args.mabs_cfg, "r")
for line in cfg_file.readlines():
    if line.startswith("training_dir"):
        training_dir = line.split("=")[-1].strip()
    elif line.startswith("registration_config"):
        reg_par_file = line.split("=")[-1].strip()

cfg_file.close()

# Define the root directories
prealign_dir = training_dir + os.sep + "prealign"
mabs_train_dir = training_dir + os.sep + "mabs-train"

# Get only the directories inside the prealign folder
subjects = list_only_dirs(prealign_dir)

# Create the folder for reports
if os.path.exists(args.output_reports_dir):
    shutil.rmtree(args.output_reports_dir)
os.mkdir(args.output_reports_dir)

# Cycles through the subjects
for subject in subjects:
    
    # Set the fixed image
    fixed_img_fn = prealign_dir + os.sep + subject + os.sep + "img.nrrd"

    # Create a report folder. Here will be saved the screenshot
    os.mkdir(args.output_reports_dir + os.sep + subject)

    # List the atlases used for this subject
    warped_atlases = list_only_dirs(mabs_train_dir + os.sep + subject)
    if "segmentations" in warped_atlases: warped_atlases.remove("segmentations")

    # Cycles through the warped atlases
    for warped_atlas in warped_atlases:

        # Set the warped atlas
        warped_img_fn = mabs_train_dir + os.sep + subject + os.sep + warped_atlas + os.sep + reg_par_file + os.sep + "img.nrrd"

        # Create the screenshot
        create_screenshot (fixed_img_fn, warped_img_fn, args.output_reports_dir + os.sep + subject + os.sep + subject + "_report_" + warped_atlas + ".png")

    # Merge single reports
    number_of_single_reports = len(os.listdir(args.output_reports_dir + os.sep + subject))
    total_report_fn = args.output_reports_dir + os.sep + subject + os.sep + "total_report_" + subject + ".png"
    for i, report_fn in enumerate(os.listdir(args.output_reports_dir + os.sep + subject)):
        total_report = plt.subplot(number_of_single_reports, 1, i+1)
        plt.axis('off')
        total_report.imshow(scipy.misc.imread(args.output_reports_dir + os.sep + subject + os.sep + report_fn))
    plt.savefig(total_report_fn, bbox_inches="tight")

    # Open total reports inside the browser
    browser = webbrowser.get(args.browser)
    browser.open(total_report_fn, new = 2)

