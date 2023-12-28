# DeepNeo: Deep Learning for neointimal tissue characterization using optical coherence tomography

Valentin Koch<sup>1,3,4</sup>, Olle Holmberg<sup>1,3</sup>, Alp Aytekin<sup>2</sup>, Masaru Seguchi<sup>2</sup>, Erion Xhepa<sup>2</sup>, Jens Wiebe<sup>2</sup>, Salvatore Cassese<sup>2</sup>, Sebastian Kufner<sup>2</sup>, Adnan Kastrati<sup>2,5</sup>, Heribert Schunkert<sup>2,5</sup>, Carsten Marr<sup>1</sup>, Julia Schnabel<sup>1,3,6</sup>, Michael Joner<sup>2,5</sup>, Philipp Nicol<sup>2,5</sup>

 <sub>
1 Helmholtz Munich - German Research Center for Environmental Health, Germany; 
2 German Heart Centre Munich, Technical University of Munich, Germany; 
3 School of Computation and Information Technology, Technical University of Munich, Germany; 
4 Munich School for Data Science, Germany; 
5 German Center for Cardiovascular Research, Partner Site Munich Heart Alliance, Germany; 
6 School of Biomedical Engineering and Imaging Sciences, King's College London, UK; 
 </sub>
 
### Introduction
We present DeepNeo, a deep learning-based algorithm, to automate the process of segmenting and characterizing neointimal tissue of stented patients (tissue that grows over the stent) in optical coherence tomography (OCT) images. OCT is a tool that provides high-resolution imaging of stented segments after PCI (percutaneous coronary intervention, a non-surgical procedure that improves blood flow to the heart, e.g. by implanting a stent). 
![DeepNeo overview](media/deepneo_graphical_abstract.png?raw=true "DeepNeo overview figure")


### Demo

We provide an online tool for researchers to quickly analyze and download results, 
#### click on the image for a video demonstration:
[![Demo Video](media/deepneo_figure8.png?raw=true)](https://www.youtube.com/watch?v=u5l_Mjlfai4)
The user interface of our webtool, which provides a user-friendly platform for the analysis of intravascular OCT images. The interface is designed with several features to facilitate accurate and efficient analysis, including an upload mask (A), which allows users to upload OCT pullback images (DICOM or .zip), a visual representation of the current OCT frame with segmentation and neointima prediction, a schematic view  of quadrants (C1) (top row represents quadrant I, bottom row quadrant IV) and neointima and lumen (C2) that provides a visual representation of the tissue characteristics, including a slider (C3) that enables users to move through the pullback. In addition, the interface includes a pullback analysis (D) that provides a detailed analysis of the OCT images and a manual correction feature (E) to correct beginning and end of the stent if necessary. The webtool also allows users to download a detailed analysis of their results and provides an information tab (F) for additional guidance. Users are required to accept the research-only use on the welcome page (G) before accessing the tool.
### Usage
To run the DeepNeo online tool, follow these steps:

#### **Step 1: Clone the Repository**
First, clone this repository to your local machine.
```git
git clone https://github.com/ValentinKoch/DeepNeo.git
```
#### **Step 2: Navigate to the App Folder**
```git
cd app
```
#### **Step 3: Create a Conda Environment**
```git
conda create --name DeepNeo --file requirements.txt
```
#### **Step 4: Activate the Environment**
```git
conda activate DeepNeo
```
#### **Step 5: Update Configuration File**
Before running the application, update the configuration file to match your system's paths. Locate the config.py file in the app folder and make the necessary adjustments to the folder paths.
#### **Step 6: Run the Application**
Within the app folder, start the DeepNeo web application by executing:
```git
python run_gradio.py
```
#### **Step 7: Access the Web Application**
After running the script, a local server will start, and a URL will be generated. Open this URL in your web browser to access the DeepNeo web application. 
