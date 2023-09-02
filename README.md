# Scrap_metal_classification
This is a project involving image processing and machine learning algorithm for scrap metal classification.

The features_extraction. py can extract area, perimeter, eccentricity, texture, colour moment and truncated fourier descriptor.

fourier_desc.py describes the extraction process of Fourier descriptors in more detail.

The lbp.py file can extract image texture features and convert them into lbp features spectrum.

quantified_fourier_desc.py file can help to quantify the lost information when using truncated fourier descriptor.

cnn_features.py is a network with the input of features extracted by features_extraction.py.

cnn.py is a more complex network which also has a better performance.

evaluate_cnn and evaluate_cnn_feature can use precision and recall to evaluate the performance of model.

SVM.py is a support vector machine algorithm for multi-classification.

It sould be noticed that the inputs of all of these algorithm are images. You can define your own classes when you use this code.
