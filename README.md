# MRI-AUTOMAP
This is an implementation of the AUTOMAP algorithm described in the following paper:
<a href="https://www.nature.com/articles/nature25988">Zhu, Bo, et al. "Image reconstruction by domain-transform manifold learning." Nature 555.7697 (2018): 487.</a>

## Sample results
![](https://raw.githubusercontent.com/chongduan/MRI-AUTOMAP/master/Img/output_new.png)

Figure 1. First row contains MRI k-space data, which is the input to the network. Second row is the direct Fourier transform of the k-space data, and finally the third row is the network-reconstructed MRI images. 

Note that the network output are blurry. This might due to the training was performed on a small dataset (~5000 cardiac MRI images).

