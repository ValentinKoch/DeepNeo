# DeepNeo: Deep Learning for neointimal tissue characterization using optical coherence tomography

Valentin Koch<sup>1,3,4</sup>, Alp Aytekin<sup>2</sup>, Masaru Seguchi<sup>2</sup>, Erion Xhepa<sup>2</sup>, Jens Wiebe<sup>2</sup>, Salvatore Cassese<sup>2</sup>, Sebastian Kufner<sup>2</sup>, Adnan Kastrati<sup>2,5</sup>, Heribert Schunkert<sup>2,5</sup>, Carsten Marr<sup>1</sup>, Julia Schnabel<sup>1,3,6</sup>, Michael Joner<sup>2,5</sup>, Philipp Nicol<sup>2,5</sup>

 <sub>
1 Helmholtz Munich - German Research Center for Environmental Health, Germany; 
2 German Heart Centre Munich, Technical University of Munich, Germany; 
3 School of Computation and Information Technology, Technical University of Munich, Germany; 
4 Munich School for Data Science, Germany; 
5 German Center for Cardiovascular Research, Partner Site Munich Heart Alliance, Germany; 
6 School of Biomedical Engineering and Imaging Sciences, King's College London, UK; 
 </sub>

### Introduction
We present DeepNeo, a deep learning-based algorithm, to automate the process of segmenting and characterizing neointimal tissue of stented patients (tissue that grows over the stent) in optical coherence tomography (OCT) images. OCT is a tool that provides high-resolution imaging of stented segments after percutaneous coronary intervention (a non-surgical procedure that improves blood flow to the heart). 

### DeepNeo provides neointimal tissue segmentation and classification on quadrant level.
![DeepNeo overview](media/deepneo_figure1.png?raw=true "DeepNeo overview figure")
A: OCT frames are divided into four 90° quadrants (Q1-Q4), rotating clockwise from 12 o’clock and individually classified to one of four classes indicated by circular line color. Vessel lumen, neointima and stent struts are annotated pixelwise. B: Representative example of homogenous, heterogenous, neoatherosclerosis and not analyzable OCT frames used in the study. C: DeepNeo architecture: A frame is given as input to a U-Net to get a segmentation mask. This allows the calculation of the center of the lumen and the division of the OCT frame into 4 quadrants at the center, which are then each resized to a size of 224x224 pixels before going through the classification network (ResNet-18). The coloured quarter-circles show the predicted class for each quadrant, line thickness indicates model certainty (thick line  = high certainty).

### Clinical Cases
![Clinical Cases](media/deepneo_figure7.png?raw=true "Clinical Cases")
3D reconstruction of neointima, lumen and stents (1) as well as 3D reconstruction of neointimal tissue prediction (2) and sample frames (3) from two clinical cases. Quantitative statistics derived from DeepNeo are provided as well. Left: 64 year-old male with PCI of RCA. OCT 12 months after PCI reveals predominately neoatherosclerotic neointima. During follow-up, the patient underwent TLR due in-stent restenosis with unstable angina . Right: 79 year-old male with PCI of LAD. OCT 12 months after PCI reveals predominantly homogenous neointima. During follow-up, no adverse events occurred. Note how neoatherosclerosis can lead to a loss of signal leading to undetected stent struts (white box in A.1 and A.2). Note the correct correct classification of uncovered stent struts as “not analyzable” (blue line in B.1 and B.2) and detection of a side-branch (white circle in B.1).

### Demo

We provide an online tool for researchers to quickly analyze and download results, 
#### click on the image for a video demonstration:
[![Demo Video](media/deepneo_figure8.png?raw=true)](https://www.youtube.com/watch?v=u5l_Mjlfai4)




