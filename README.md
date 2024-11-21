![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

[![Machine Learning Client CI](https://github.com/software-students-fall2024/4-containers-fantastic-four/actions/workflows/ml-client.yml/badge.svg?branch=main)](https://github.com/software-students-fall2024/4-containers-fantastic-four/actions/workflows/ml-client.yml)

![event-logger](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/event-logger.yml/badge.svg)

# Containerized App Exercise

Our app PlantifyAI leverages Machine Learning models for plant species identification. This system allows users to identifies the species of a plant from the uploaded photo.

This app demonstrates how to perform flower classification using **transfer learning** and the **ResNet50** model. We fine-tune a pre-trained ResNet50 model on the Flowers102 dataset, which consists of 102 species of flowers. The final goal is to achieve **>95% accuracy** on the validation set. 

## Teammates

[Hang Yin](https://github.com/Popilopi168)

[Jessica Xu](https://github.com/Jessicakk0711)

[Alexandra Mastrangelo](https://github.com/alexandramastrangelo)

[Natalie Trump](https://github.com/nht251)
## Installation

To set up this project locally, follow these steps:

1. **Clone the repository** (or download the files):
    ```bash
    git clone https://github.com/software-students-fall2024/4-containers-fantastic-four.git
    cd <repository-folder>
    ```

2. **Create a virtual environment**:
    - make sure to create `.env` file under both `web-app` and `machine-learning-client` folder
    - create a virtual environment
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

5. **Build using Docker**:
    ```bash
    docker-compose up --build
    ```

    - This will start the Flask app inside the container and map it to port 5001 on your host machine. You can access the application at http://127.0.0.1:5001/. The machine learning client will map to port 3000.


## Machine Learning Client Usage:
The full training script is also included in the project, and the trained model is saved as `flower_classification_resnet.pth`.

```python
# Example of training the model in the main function
python train.py
```

### Model Performance

After training and fine-tuning, the model achieves **>95% accuracy** on the validation set, demonstrating the power of transfer learning using ResNet50 for this flower classification task.

### Prediction Using the `main()` Function

To predict the flower species for a given image, you can use the `main()` function in `app.py`.

### Example Usage:

```python
def main(image_path):
    flower_names = load_flower_names()  # Load flower name mapping
    model = load_model()  # Load the pre-trained model
    plant_name = predict_plant(image_path, model, flower_names)
    print(f"Predicted plant: {plant_name}")
```

```python
# Example usage with a test image path
main('data/flowers-102/jpg/image_08004.jpg')  # Adjust the image path as needed
```

## Project Task Board
We are actively tracking our progress using a [Task Board](https://github.com/orgs/software-students-fall2024/projects/127). 
