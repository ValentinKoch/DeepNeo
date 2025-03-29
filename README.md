# DeepNeo: Deep Learning for neointimal tissue characterization using optical coherence tomography

**Valentin Koch**<sup>1,2,3</sup>, **Olle Holmberg**<sup>1,2,4</sup>, **Edna Blum**<sup>5</sup>, **Ece Sancar**<sup>1,2</sup>, **Alp Aytekin**<sup>5</sup>, **Masaru Seguchi**<sup>5</sup>, **Erion Xhepa**<sup>5</sup>, **Jens Wiebe**<sup>5</sup>, **Salvatore Cassese**<sup>5</sup>, **Sebastian Kufner**<sup>5</sup>, **Thorsten Kessler**<sup>5,6</sup>, **Hendrik Sager**<sup>5,6</sup>, **Felix Voll**<sup>5</sup>, **Tobias Rheude**<sup>5</sup>, **Tobias Lenz**<sup>5</sup>, **Adnan Kastrati**<sup>5,6</sup>, **Heribert Schunkert**<sup>5,6</sup>, **Julia A. Schnabel**<sup>1,2,7</sup>, **Michael Joner**<sup>5,6</sup>, **Carsten Marr**<sup>1</sup>, **Philipp Nicol**<sup>5</sup>

<small> 1. Helmholtz Munich - German Research Center for Environmental Health, Munich, Germany </small> 
2. School of Computation and Information Technology, Technical University of Munich, Munich, Germany  
3. Munich School for Data Science, Munich, Germany  
4. Helsing GmbH, Munich, Germany  
5. German Heart Centre Munich, Technical University of Munich, Munich, Germany  
6. German Center for Cardiovascular Research, Partner Site Munich Heart Alliance, Munich, Germany  
7. School of Biomedical Engineering and Imaging Sciences, King's College London, London, UK


 
### Introduction
We present DeepNeo, a deep learning-based algorithm, to automate the process of segmenting and characterizing neointimal tissue of stented patients (tissue that grows over the stent) in optical coherence tomography (OCT) images. OCT is a tool that provides high-resolution imaging of stented segments after PCI (percutaneous coronary intervention, a non-surgical procedure that improves blood flow to the heart, e.g. by implanting a stent). 
![DeepNeo overview](media/deepneo_graphical_abstract.png?raw=true "DeepNeo overview figure")


### Demo

We provide a tool for researchers to quickly analyze and download results, 
#### click on the image for a video demonstration:
[![Demo Video](media/deepneo_figure8.png?raw=true)](https://www.youtube.com/watch?v=u5l_Mjlfai4)
The user-friendly interface is designed with several features to facilitate accurate and efficient analysis, including an upload mask (a), which allows users to upload OCT pullback images (DICOM or .zip), a visual representation of the current OCT frame with segmentation and neointima prediction (b), a schematic view of quadrants (c) (top row represents quadrant I, bottom row quadrant IV) and neointima and lumen (d) that provides a visual representation of the tissue characteristics, including a slider that enables users to move through the pullback. In addition, the interface includes a pullback analysis (e) that provides a detailed analysis of the OCT images and a manual correction feature (f) to correct beginning and end of the stent if necessary. The webtool also allows users to download a detailed analysis of their results and provides an information tab (g) for additional guidance. Users are required to accept the research-only use on the welcome page (h) before accessing the tool. 

### Usage
To run the DeepNeo tool, follow these steps:

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


### Models & Contact
We are very happy to provide models, please request them at [Zenodo](https://zenodo.org/records/14556455) and contact Valentin Koch (valentin.koch@helmholtz-munich.de), Carsten Marr (carsten.marr@helmholtz-munich.de), or Michael Joner (joner@dhm.mhn.de).
