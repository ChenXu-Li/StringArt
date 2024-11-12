# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def matrix2pos(matrix):
	flat_indices = np.argpartition(matrix.flatten(), -config.NAIL_NUM)[-config.NAIL_NUM:]
	top_coords = np.unravel_index(flat_indices, matrix.shape)
	top_coords = list(zip(top_coords[0], top_coords[1]))
	return top_coords
def prepare_plot(origImage, origMask, predMask,ProbabilityMatrix,top_coords):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	ax[3].imshow(origImage)
	ax[4].imshow(ProbabilityMatrix)

	x_coords, y_coords = zip(*top_coords)
	ax[3].scatter(y_coords, x_coords, color='red', s=1, label='Top Points')  # Red points, size 50
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	ax[3].set_title("Nailpos Img")
	ax[4].set_title("Probability Matrix")
	
	# set the layout of the figure and display it
	figure.tight_layout()
	#figure.show()
	plt.show()

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		# image = cv2.imread(imagePath,0)
		# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# image = image/255.0
		# 从磁盘加载图像，读取为灰度图
		image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
		print("Image shape:", image.shape)
		image= np.expand_dims(image, axis=2)
		# print("Image shape:", image.shape)
		# image = np.stack((image,) * 1, axis=-1)
		print("Image shape:", image.shape)
		# 检查图像是否成功加载
		if image is None:
			raise FileNotFoundError(f"Image not found at path: {imagePath}")
        # 将图像归一化到 [0, 1] 范围
		image = image / 255.0


		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))
		
        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		# 进行转置
		if image.ndim == 3:
			image = np.transpose(image, (2, 0, 1))  # (C, H, W)
		elif image.ndim == 2:
			image = np.expand_dims(image, axis=0)  # 变为 (1, H, W)
		print("Image shape:", image.shape)
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).float().to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		top_coords = matrix2pos(predMask)
		ProbabilityMatrix = (predMask.copy()*255).astype(np.uint8)

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask,ProbabilityMatrix,top_coords)
		
# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)