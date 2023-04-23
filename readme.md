<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Pet Image Segmentation
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-customobjectdetectiontfod2.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1APGmivgM0v6Pwl5D8ptLdIwMaXutgz8t)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Modelling__](#e-modelling)
    - [__(D) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)
## __Brief__ 

### __Project__ 
- This is an __Object Detection__ (a subtask of classification) project as image __object Detection Task__ that uses the  [__Covid Mask Dataset__](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format?select=labeled_mask_dataset) to __detect the mask objcet__ from image.
- The __goal__ is build a deep learning object detection model that accurately __detects the covid mask__ from images.
- The performance of the model is evaluated using several __metrics__ loss and accuracy metrics.

#### __Overview__
- This project involves building a deep learning model to detect the covid mask from images. The dataset contains 1370 images  2 classe about covid mask.  The models has been trained with google tensorflow object detection 2 api, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, tensorflow, TFOD2 api from tensorflow.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-customobjectdetectiontfod2.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/18ZKfjtuWeI9eSwzdYkhRBUjfGXnqgtv1#scrollTo=p1KlES7FzMkA"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/CustomObjectDetectionTFOD2/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1APGmivgM0v6Pwl5D8ptLdIwMaXutgz8t"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    -  __Detects covid mask__ from images.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-customobjectdetectiontfod2.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-customobjectdetectiontfod2.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Covid Mask Dataset__](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format?select=labeled_mask_dataset) from tensorflow dataset api.
- The dataset contains 1370 images  2 classe about covid mask.
- The dataset contains the following features:

#### Problem, Goal and Solving approach
- This is an __image seggmentation__ problem  that uses the   [__Covid Mask Dataset__](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format?select=labeled_mask_dataset)  to __detecting covid masks__ from images.
- The __goal__ is build a deep learning object detection model that accurately __detects the masks__ from images.
- __Solving approach__ is that using the supervised deep learning models. Fine tuning approach implemented on ssd_mobilenet_v2 state of art model for detecting the mask object from image. TFOD2 api is used for training phase. 

#### Study
The project aimed segmentating the pets using deep learning model architecture. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries. Installing TFOD2 api and related requirements (it can be complex, but just run the codes it will be handled)
- __(B) Dataset__: Downloading and loading the dataset. Preparing the dataset via tensorflow dataset api. Configurating the dataset performance and related pre-processes. 
- __(C) Preprocessing__: Configurating the model of TFOD2 api.
- __(D) Modelling__:
  - Selecting the state of art model for object detection as ssd_mobilenet_v2.
  - Configurating the training setting of the state of art model.
  - Starting the traning phase of the state of art model for fine-tuning using through TFOD2 api.
- __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final object detection model is __ssd_mobilenet_v2__ because of the results and less complexity.
  -  Fine tuned ssd_mobilenet_v2 model Results
        <table><tr><th>Object Detection Results </th><th></th></tr><tr><td>
    | Metric name                            | Score     |
    |----------------------------------------|-----------|
    | DetectionBoxes_Precision/mAP           | 0.813305  |
    | DetectionBoxes_Precision/mAP@.50IOU    | 0.996689  |
    | DetectionBoxes_Precision/mAP@.75IOU    | 0.980285  |
    | DetectionBoxes_Precision/mAP (small)   | 0.800000  |
    | DetectionBoxes_Precision/mAP (medium)  | 0.741907  |
    | DetectionBoxes_Precision/mAP (large)   | 0.826483  |
    | DetectionBoxes_Recall/AR@1             | 0.754275  |
    | DetectionBoxes_Recall/AR@10            | 0.849182  |
    | DetectionBoxes_Recall/AR@100           | 0.851497  |
    | DetectionBoxes_Recall/AR@100 (small)   | 0.800000  |
    | DetectionBoxes_Recall/AR@100 (medium)  | 0.785000  |
    | DetectionBoxes_Recall/AR@100 (large)   | 0.861485  |
    | Loss/localization_loss                 | 0.030248  |
    | Loss/classification_loss               | 0.074692  |
    | Loss/regularization_loss               | 0.123254  |
    | Loss/total_loss                        | 0.228193  |

    </td></tr></table>

## Details

### Abstract
-  [__Covid Mask Dataset__](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format?select=labeled_mask_dataset) is used to detect covid mask object from images.  The dataset contains 1370 images with 2 classes (mask, non-mask) about covid mask object from images. The problem is a supervised learning task as object detection (a subtask of classification). The goal is detecting  the covid mask object using through supervised state of art deep learning algorithms or related training approachs of pretrained state of art models via TFOD2 api from tensorflow. The study includes creating the environment, getting the data, preprocessing the data,  setting the TFOD2 api, setting the model and its configurations, settings the tfod2 api for realated processes, modelling the data, saving the results, extracting the model, deployment as demo app. Training phase of the models implemented through tensorflow callbacks. Fine tuning approach is implemented on state of art model using tfod2 api . Selected the basic and more succesful when comparet between other models  is  fine_tuned_ssd_mobilenet_v2 object detection model.__fine_tuned_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8__ model has __0.2281__ loss , __0.8133__ precision mAP,  other metrics (detailed object detection metrics provided by TFOD2 api) are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── examples
│   └── fine_tuned_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
├── env
│   └── env_installation.md
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/fine_tuned_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8:
    - Fine tuned ssd_mobilenet_v2 model.
  - demo_app/examples
    - Example cases to test the model.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    

### Explanation of the Study
#### __(A) Dependencies__:
  - There are third part dependencies such as tfod2 api and related requirements. It can be complex but just follow the codes, additionally you can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from tensoflow.
#### __(B) Dataset__: 
  - Downloading the [__Covid Mask Dataset__](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format?select=labeled_mask_dataset)  via tensorflow dataset api. 
  - The dataset contains 1370 images  2 classe about covid mask.
  - Preparing the image dataset for TFOD2 api.
  - Creating the tensorflow dataset object then configurating the state of art object detection models.
#### __(C) Modelling__: 
  - The processes are below:
    - Archirecture
      - Fine-Tuned ssd_mobilenet_v2 
    - Training
      - Fine Tuning the state of art models via tensorflow TFOD2. Setting the configurations corresponding for object detection task. 
    - Evaluating and classification results
          <table><tr><th>Object Detection Results </th><th></th></tr><tr><td>
      | Metric name                            | Score     |
      |----------------------------------------|-----------|
      | DetectionBoxes_Precision/mAP           | 0.813305  |
      | DetectionBoxes_Precision/mAP@.50IOU    | 0.996689  |
      | DetectionBoxes_Precision/mAP@.75IOU    | 0.980285  |
      | DetectionBoxes_Precision/mAP (small)   | 0.800000  |
      | DetectionBoxes_Precision/mAP (medium)  | 0.741907  |
      | DetectionBoxes_Precision/mAP (large)   | 0.826483  |
      | DetectionBoxes_Recall/AR@1             | 0.754275  |
      | DetectionBoxes_Recall/AR@10            | 0.849182  |
      | DetectionBoxes_Recall/AR@100           | 0.851497  |
      | DetectionBoxes_Recall/AR@100 (small)   | 0.800000  |
      | DetectionBoxes_Recall/AR@100 (medium)  | 0.785000  |
      | DetectionBoxes_Recall/AR@100 (large)   | 0.861485  |
      | Loss/localization_loss                 | 0.030248  |
      | Loss/classification_loss               | 0.074692  |
      | Loss/regularization_loss               | 0.123254  |
      | Loss/total_loss                        | 0.228193  |

      </td></tr></table>
  - Saving the project and demo studies.
    - trained model __fine_tuned_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8__ as tensorflow (keras) saved_model format.

#### __(D) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is detecting covid mask object from images.
    - Usage: upload or select the image for detecting then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-customobjectdetectiontfod2.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

