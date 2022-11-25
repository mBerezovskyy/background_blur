# background_blur

>**Own implementation of the most popular selfie algorithm with**
>**same neural network as in Google Pixel 3XL phone**

_Created using opencv and DeeplabV3+ with Xception71 backbone_


# **_Instalation and usage_**

```
run pip install -r requirements.txt
choose photo of arbitrary resolution and aspect ratio and save it in root project directory
substitute name of the file in line 7
run main.py file
```

# **_Results_**
![result1](https://user-images.githubusercontent.com/28948578/204031072-3cc3b203-527c-4eef-ad70-cbdef9c20b96.png)
![result2](https://user-images.githubusercontent.com/28948578/204031077-05f52565-e3db-4e5b-bb56-d2ef6824ec5a.png)


# **_Future improvements:_**
- Eliminate bluring of random blobs that do not represent body shape
- Eliminate bluring parts of body as shoulder part in image 2
- Make transition from segmented body to blured background smoother
- Smooth detected body contour
- Try different segmentation nn architectures for example Unet
