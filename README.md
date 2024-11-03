AIR-CANVAS WITH MACHINE LEARNING

A Computer Vision project utilizing OpenCV and Machine Learning with MediaPipe

PROJECT OVERVIEW

Have you ever wanted to sketch in the air, using only the movement of your fingers? With this project, you can! This Air Canvas allows you to draw your imagination onto a digital canvas simply by moving your hand in the air. By tracking specific landmarks on your hand, this project creates a unique and interactive drawing experience—perfect for anyone looking to showcase their skills in computer vision and machine learning.
This project leverages OpenCV for computer vision techniques and MediaPipe for landmark detection, making it a great addition to any machine learning portfolio.

KEY FEATURES

Real-time Hand Landmark Detection: Utilizes MediaPipe to identify key points on the hand and track finger movements.
Air Drawing Mechanism: Draw by simply moving your finger over a designated area, which will record and display your sketches on the canvas.
Interactive Color Selection: Choose colors by moving your hand over color buttons, or clear the canvas with a simple hand gesture.
Python & OpenCV Implementation: Uses Python for its comprehensive libraries and easy syntax, though the basics can be applied to any OpenCV-supported language.
Working Principle
This project’s core is the hand landmark detection and tracking algorithm using MediaPipe. The steps below summarize the process:

ALGORITHM STEPS

1.Frame Capture & Color Conversion: Begin by capturing video frames and convert them to the HSV color space. This color space is more suitable for detecting colors in OpenCV.

2.Canvas Preparation: Set up a blank canvas and create virtual color buttons for ink selection.

3.Hand Detection Configuration: Configure MediaPipe to detect and track only one hand for streamlined interaction.

4.Landmark Detection: Pass each frame through the MediaPipe hand detector to identify hand landmarks in real-time.

5.Coordinate Tracking: Locate the coordinates of the fingertip (forefinger) and store them for consecutive frames. This helps draw a continuous line on the canvas as the finger moves.

6.Drawing on Canvas: Using the stored coordinates, draw on both the frame and the canvas, allowing the air-drawing effect to come to life.

REQUIREMENTS

Python 3.x
OpenCV
NumPy
MediaPipe

This Air Canvas project is a fun and dynamic way to explore computer vision and machine learning concepts. Not only is it visually engaging, but it also demonstrates the practical application of MediaPipe and OpenCV, making it an impressive addition to any portfolio.
