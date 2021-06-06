# Image-Colorization
Xuan Dong, Weixin Li, Xiaojie Wang, Yunhong Wang.In AAAI,2019<br><br>

This is the implementation code of AAAI2019's paper "Learning a Deep Convolutional Network for Coloration in Monochrome-Color Dual-Lens System".The example coloring result of gray image is shown in the figure below.<br>
![图片](https://user-images.githubusercontent.com/84729271/120930077-555c7100-c71e-11eb-8d52-6cc9be3851dd.png)<br><br>
Clone the repository.<br>
`git clone https://github.com/bupt-wx/AAAI2020-Image-Colorization_of_dx.git`<br>
Required environment version information.<br>
`Tensorflow 1.11; Python 3.6`<br><br>
The algorithm is divided into rough coloring and color correction.<br>
You can test this project by using the following commands and using the images in the Sample_input folder.It should be noted that the algorithm uses the Ycbcr color space, and the pre-processing and post-processing of the algorithm requires converting the color space of the image.<br>
The first step-rough colorization, using test files in RoughColorization folder.The test command is as follows:<br>
`python rough_colorization_test.py -fpath test_input_file_path -outpath test_output_file_path`<br>
Please replace "test_input_file_path" with the input image path to be tested and "test_output_file_path" with the output image path after testing.<br>
The second step-color correction, using test files in Correction folder.The test command is as follows::<br>
`python colorization_correction_test.py -rpath rough_colorization_result_file_path -gpath guided_image_file_path -outpath test_output_file_path`<br>
Please replace "rough_colorization_result_file_path" with the results obtained by the first step of rough coloring, "guided_image_file_path" with the input grayscale image and "test_output_file_path" with the output image path after testing.The results should match the images in the Sample_out folder.<br>
