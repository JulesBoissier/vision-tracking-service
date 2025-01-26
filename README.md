# Vision Tracking Service

An ML-Powered Gaze tracking and point of regard estimation tool wrapped in a FastAPI-based web-server.

### High-Level Architecture and Key Components

<div align="center">
  <img src="assets/architecture-diagram.png" alt="High-Level Architecture Diagram" width="600"/>
  <p><strong>Fig 1.</strong> High-Level Architecture Diagram of Key Components of the Vision Tracking Service</p>
</div>

**Vision Tracking Engine:** The central orchestration object leveraging composition of the three other key component to manage calibration profiles and predict point of regard.

**Gaze Predictor:** A wrapper around the L2CSNet model which, given an image, returns a prediction for the Gaze Vector of the person in the picture, along with coordinates of their eyes.

**Calibration Agent:** Runs interpolation to predict the Point of Regard, given a Gaze vector by leveraging pre-recorded calibration points.

**Calibration Profile Store:** An SQLAlchemy-based database handler to store and retrieve Calibration Profiles.


### Usage

1. Download the pre-trained Gaze360 Model from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd) and drop the pickle file into the models directory.

2. Run docker-compose to build the container and persistent volume.

3. Optionally, load an existing profile from the persistent volume.

```bash
curl -X POST "http://127.0.0.1:8000/load_profile?profile_id=1"
```

4. Curl a few calibration points to the /cal_point end-point to configure a calibration profile. 

```bash
curl -X POST "http://127.0.0.1:8000/cal_point" \
     -F "file=@image.jpg" \
     -F "x=0" \
     -F "y=100"
```

5. Optionally save the profile to re-use later.

```bash
curl -X POST "http://127.0.0.1:8000/save_profile?name=my_profile"
```

6. Predict the Point of Regard based on a new image.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@new_image.jpg"
```