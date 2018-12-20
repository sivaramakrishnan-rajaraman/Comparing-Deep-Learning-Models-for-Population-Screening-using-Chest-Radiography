# Comparing-Deep-Learning-Models-for-Population-Screening-using-Chest-Radiography

Feel free to use the attached notebooks for your own model and data. The repository also includes Matlab scripts to use AlexNet and extract features from the underlying data. Kindly cite the publication as these codes are part of this published work:

### Rajaraman S, Antani SK, Candemir S, Xue Z, Abuya J, Kohli M, Alderson P, Thoma GR. Comparing deep learning models for population screening using chest radiography. Proc. SPIE 10575, Medical Imaging 2018: Computer-Aided Diagnosis, 105751E (27 February 2018).

# Prerequisites: 

Keras>=2.2.0

Tensorflow-GPU>=1.9.0

OpenCV>=3.3

# Goal
In this study, we evaluated the performance of customized and pretrained CNN based DL models for population screening using frontal chest X-rays (CXRs). We also experimentally determined the optimal layer in the pre-trained models for extracting the features to aid in improved Tuberculosis (TB) detection. The results demonstrate that pretrained CNNs are a promising feature extracting tool for medical imagery including the automated diagnosis of TB from chest radiographs but emphasize the importance of large data sets for the most accurate classification. Please find included the published paper for more details on the methods, results and discussion. 

# Data Availability
TB CXR Image Datasets: The following de-identified image data sets of CXRs are available to the research community. Both sets contain normal as well as abnormal x-rays, with the latter containing manifestations of tuberculosis.https://openi.nlm.nih.gov/faq.php#faq-tb-coll

Montgomery County X-ray Set: X-ray images in this data set have been acquired from the tuberculosis control program of the Department of Health and Human Services of Montgomery County, MD, USA. This set contains 138 posterior-anterior x-rays, of which 80 x-rays are normal and 58 x-rays are abnormal with manifestations of tuberculosis. All images are de-identified and available in DICOM format. The set covers a wide range of abnormalities, including effusions and miliary patterns. The data set includes radiology readings available as a text file. http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip


Shenzhen Hospital X-ray Set: X-ray images in this data set have been collected by Shenzhen No.3 Hospital in Shenzhen, Guangdong providence, China. The x-rays were acquired as part of the routine care at Shenzhen Hospital. The set contains images in JPEG format. There are 340 normal x-rays and 275 abnormal x-rays showing various manifestations of tuberculosis. http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

For additional information about these datasets, please refer to  our paper https://lhncbc.nlm.nih.gov/system/files/pub9356.pdf

Kenya Dataset is a private dataset and will be made publicly available later. India dataset is a publicly available dataset at https://sourceforge.net/projects/tbxpredict/files/data/.  

# ROI Segmentation

The CXRs contain regions other than the lungs that do not contribute to diagnosing TB. We used an algorithm based on anatomical atlases to automatically detect the lung ROI and cropped them to the size of the bounding box and resampled to 224 X 224 and 299 X 299 pixel dimensions to suit the input requirements of customized and pretrained CNN models . The source code for the same can be found at https://ceb.nlm.nih.gov/proj/tb/Segmentation_Module_Version_2017_11_04.zip.

# Performance Evaluation

We evaluated the performance of different types of custom CNNs including (i) Sequential CNN; (ii) AlexNet; (iii) VGG16; (iv) VGG19; (v) Xception; and (vi) ResNet50 in detecting TB. We evaluated the performance of these CNNs in terms of the following performance metrics: (i) accuracy; (ii) AUC.
