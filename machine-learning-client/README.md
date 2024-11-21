# Flower Classification with Transfer Learning using ResNet50

This project demonstrates how to perform flower classification using **transfer learning** and the **ResNet50** model. We fine-tune a pre-trained ResNet50 model on the Flowers102 dataset, which consists of 102 species of flowers. The final goal is to achieve **>95% accuracy** on the validation set. 

The project includes an inference function for classifying flowers based on input images, which can be found in `app.py` as `main()`.

## Project Overview

### Key Features:
- **Transfer Learning with ResNet50**: We leverage a pre-trained **ResNet50** model, trained on ImageNet, and fine-tune it to classify 102 species of flowers.
- **Achieving High Accuracy**: Through proper model training and validation, we aim for an accuracy of **>95%** on the validation set.
- **102 Species Classification**: The model is trained to classify 102 different flower species from the **Flowers102 dataset**.
- **Easy-to-use Prediction Function**: The main function for making predictions is implemented in `app.py` and is easily callable with an image input.

## Dataset

The dataset used in this project is the **Flowers102 dataset**, which contains **102 flower categories** with around **8,000 images**. The images are organized in a flat directory structure with all the images stored in `flowers-102/jpg`. The dataset also includes the `flower_to_name.json` file, which maps class labels to flower species names.

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository** (or download the files):
    ```bash
    git clone https://github.com/software-students-fall2024/4-containers-fantastic-four.git
    cd machine-learning-client
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv flower-env
    source flower-env/bin/activate  # On Windows use `flower-env\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the dataset**:
    - Download the **Flowers102 dataset** and place the `jpg` folder and `flower_to_name.json` in the `data/flowers-102` directory.
    - The directory structure should look like this:
    ```
    data/
      flowers-102/
        jpg/
          image_00001.jpg
          image_00002.jpg
          ...
        flower_to_name.json
    ```

## Training the Model

### Steps:
1. **Dataset Preprocessing**: We apply transformations to resize and normalize the images before passing them into the model.
2. **Transfer Learning**: We fine-tune the ResNet50 model, pre-trained on ImageNet, by replacing its final layer to match the number of classes in the Flowers102 dataset (102 classes).
3. **Fine-Tuning**: We freeze the earlier layers of the model (using feature extraction) and train only the final fully connected layer initially. Then, we unfreeze all layers and fine-tune the entire model.
4. **Validation**: We evaluate the model on the validation set after each epoch to track performance.

### Training Code:
The full training script is included in the project, and the trained model is saved as `flower_classification_resnet.pth`.

```python
# Example of training the model in the main function
python train.py
```

## Model Performance

After training and fine-tuning, the model achieves **>95% accuracy** on the validation set, demonstrating the power of transfer learning using ResNet50 for this flower classification task.

## Prediction Using the `main()` Function

To predict the flower species for a given image, you can use the `main()` function in `app.py`.

### Example Usage:

```python
def main(image_path):
    flower_names = load_flower_names()  # Load flower name mapping
    model = load_model()  # Load the pre-trained model
    plant_name = predict_plant(image_path, model, flower_names)
    print(f"Predicted plant: {plant_name}")
```

# Example usage with a test image path
```main('data/flowers-102/jpg/image_08004.jpg')  # Adjust the image path as needed
```


### How to Build and Run the Docker Container

#### Build the Docker Image:
1. Navigate to the directory where your `Dockerfile` and `requirements.txt` are located.
2. Run the following command to build the Docker image:

```bash
docker build -t plant-identifier-app .
```

#### Run the Docker Container:
Once the image is built, you can run the container with the following command:

```bash
docker run -p 3000:3000 plant-identifier-app
```

This will start the Flask app inside the container and map it to port 3000 on your host machine. You can access the application at http://localhost:3000.


### Additional Notes:

Ensure that MongoDB is running and accessible from within the container. If you're using a local MongoDB instance, you might need to configure the MongoDB URI in the .env file (or hardcode it) so that the Flask app can connect to it.

Example .env file:

```env
SECRET_KEY=supersecretkey
MONGO_URI=mongodb://localhost:27017  # Adjust to your MongoDB URI
MONGO_DBNAME=plant_identifier_db
FLASK_PORT=3000
```

