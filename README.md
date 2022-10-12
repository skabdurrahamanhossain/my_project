# my_project


<!-- 
                    SOFTWARE REQUREMENTS :
                                            Python:
                                            Python 3
                                            Libraries
                                            Numpy
                                            Scipy
                                            Playsound
                                            Dlib
                                            Imutils
                                            opencv, etc.
                                            Operating System
                                            Windows or Ubuntu.
                                            Google Colab Notebook.

                    HARDWARE REQUIREMENTS :
                                            Laptop with basic hard ware
                                            Web cam



            Python 3:

                    Playsound:
                                Python 3 is a newer version of the Python programming language which was
                                released in December 2008. This version was mainly released to fix problems that
                                exist in Python 2. The nature of these changes is such that Python 3 was
                                incompatible with Python 2. It is backward incompatible.
                            Dlib:
                                Dlib is a toolkit for making real world machine learning and data analysis
                                applications in C++. While the library is originally written in C++, it has
                                good, easy to use Python bindings. I have majorly used dlib for face detection
                                and facial landmark detection.

                            NumPy:
                                NumPy is a library for the Python programming language, adding support for
                                large, multi-dimensional arrays and matrices, along with a large collection of
                                high-level mathematical functions to operate on these arrays.

                            SciPy:
                                SciPy is a free and open-source Python library used for scientific computing and
                                technical computing. SciPy contains modules for optimization, linear algebra,
                                integration, interpolation, special functions, FFT, signal and image processing,
                                ODE solvers and other tasks common in science and engineering.
                                SciPy is a scientific computation library that uses NumPy underneath. SciPy stands
                                for Scientific Python. It provides more utility functions for optimization, stats and
                                signal processing.


                            playsound is a “pure Python, cross platform, single function module with no
                            dependencies for playing sounds.” With this module, you can play a sound file
                            with a single line of code: from playsound import playsound
                            playsound('myfile.wav').
                            Imuties:
                                A series of convenience functions to make basic image processing functions such
                                as translation, rotation, resizing, skeletonization, and displaying Matplotlib images
                                easier with OpenCV and both Python 2.7 and Python 3.
                            OpenCV:
                                OpenCV (Open Source Computer Vision Library) is an open source computer
                                vision and machine learning software library. OpenCV was built to provide a
                                common infrastructure for computer vision applications and to accelerate the
                                use of machine perception in the commercial products. Being a BSD-licensed
                                product, OpenCV makes it easy for businesses to utilize and modify the code.
                            Google colab Notebook:
                                Colaboratory, or “Colab” for short, is a product from Google Research. Colab
                                allows anybody to write and execute arbitrary python code through the
                                browser, and is especially well suited to machine learning, data analysis and
                                education.



                            EYE ASPECT RATIO - THRESHOLD VALUE
                The ratio of distances between the vertical eye landmarks and the
                distances between the horizontal eye landmarks
                The concept behind EAR is : The return value of the eye aspect ratio
                will be almost constant when the eye is open. However the value
                willdecrease rapidly towards zero during a blink. If the eye is
                closed, theeye aspect ratio will again remain almost constant, but
                will be significantly smaller compared to the ratio when the eye is
                open
                If the EAR value of the driver is less than the Threshold value than
                that means the driver is in drowsiness state and if it is more than the
                Threshold value than the driver is in normal condition.
                The alarm will alert the driver only when if ear value is less than
                threshold value.
                Final Eye Aspect Ratio:
                ear=(A=B)/(2.0*C)

 -->