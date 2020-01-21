
#【！】please refer to https://www.jianshu.com/p/60ce8b93b806
#【！】weight map is used for implict contour learning of Unet-w and InvUnet-w

from skimage import io
import matplotlib.pyplot as plt
from skimage import measure,color
import cv2

# 【0】 load test image
gt = io.imread('bin_img.bmp')
gt = 1 * (gt >0)
plt.figure(figsize = (10,10))
plt.imshow(gt)
plt.show()

# 【1】Calculate pixel frequency of cells and background
c_weights = np.zeros(2)
c_weights[0] = 1.0 / ((gt == 0).sum())
c_weights[1] = 1.0 / ((gt == 1).sum())

# 【2】normalization
c_weights /= c_weights.min() 

# 【3】get class_weight map(cw_map)
cw_map = np.where(gt==0, c_weights[0], c_weights[1])
plt.figure(figsize = (10,10))
plt.imshow(cw_map)
plt.show()

#【4】connected component analysis and colorization
cells = measure.label(gt, connectivity=2)
cells_color = color.label2rgb(cells , bg_label = 0,  bg_color = (0, 0, 0)) 
plt.figure(figsize = (10,10))
plt.imshow(cells_color)
plt.show()

#【5】calculate distance weight map (dw_map)
w0 = 10
sigma = 5
dw_map = np.zeros_like(gt)
maps = np.zeros((gt.shape[0], gt.shape[1], cells.max()))
if cells.max()>=2:
    for i in range(1, cells.max() + 1):
        maps[:,:,i-1] =  cv2.distanceTransform(1- (cells == i ).astype(np.uint8), cv2.DIST_L2, 3)
    maps = np.sort(maps, axis = 2)
    d1 = maps[:,:,0]
    d2 = maps[:,:,1]
    dis = ((d1 + d2)**2) / (2 * sigma * sigma)
    dw_map = w0*np.exp(-dis) * (cells == 0)
plt.figure(figsize = (10,10))
plt.imshow(dw_map, cmap = 'jet')
plt.show()

# 【6】 add cw_map and dw_map, and get the final weight map
weight_map = cw_map + dw_map
plt.figure(figsize = (20,20))
plt.imshow(weight_map, cmap = 'jet')
plt.colorbar()
plt.show()
