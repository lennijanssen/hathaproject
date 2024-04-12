# Hatha Support - The Yoga Pose Trainer

Yoga Pose Detector is a Yoga pose correction tool that recognizes the Yoga pose you do from a live feed, evaluates performance and suggests a body part to correct. It is trained on 5 poses, of which 3 work very well and the other two depend on environmental factors (like the background). The project was created as the final project for the LeWagon Data Science Bootcamp in two weeks by a group of three people. Example screenshots of deployment in Streamlit Cloud can be found in the corresponding folder of the repository. The programming language used was python. All packages used can be found in the requirement.txt file. The app can be deployed as is on local Streamlit.

[Click on this Google presentation link to find out how it works](https://docs.google.com/presentation/d/1UfZ09jS40f0SCPVXwRrNYAfss9Pa58725VnYuifE7KU/edit?usp=sharing)

## Getting started with Yoga practice

**with installed git**

1.  Clone the repository from Github
    ```sh
    git clone https://github.com/masaki-norton/hathaproject.git
    ```
2.  Install the required dependencies
    ```sh
    pip install -r requirements.txt
    ```
3.  Run the app with Streamlit
    ```sh
    streamlit run app_main.py
    ```

Online app home (if not over resource limit): https://hathaproject-trainer.streamlit.app/
   
## Built With
- [Tensorflow Movenet](https://www.tensorflow.org/hub/tutorials/movenet) - Human pose estimation
- [Keras](https://keras.io/) - Neural Network (via Tensorflow)
- [Joblib](https://joblib.readthedocs.io/) - Lightweight pipeline structure for calculations
- [WebRTC](https://webrtc.org/) - Online livestream
- [OpenCV](https://opencv.org/) - Overlay of livestream
- [Streamlit](https://streamlit.io/) - Deployment


## Acknowledgements
The project was inspired by weekly Yoga practice with our Yoga teacher Mika Saito, who was also the model for our best poses we used as a reference for the correction mechanism.
**Website**: www.mikayogaacro.com , **Instagram**: (www.instagram.com/mikayoga.acro)

## Team Members
- Lennart Janssen  (www.linkedin.com/in/lennijanssen) (https://github.com/lennijanssen)
- Masaki Norton    (www.linkedin.com/in/masaki-norton) (https://github.com/masaki-norton)
- Maria Miranda    (https://github.com/mirmachr)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is open source. Out of curiosity, do let us know if you use it for anything.
