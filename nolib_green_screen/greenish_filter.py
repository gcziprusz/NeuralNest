#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/imgs/greenML.png
#curl -LO https://github.com/mlittmancs/great_courses_ml/raw/master/imgs/forest.jpg

import numpy as np
from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt

img = keras.utils.load_img("greenML.png")

plt.imshow(img)
plt.show()

#we trim the edges of the image.
arr = image.img_to_array(img)
# Trim off edges
arr = arr[:697,:]
plt.imshow(image.array_to_img(arr,scale=False))
plt.show()

# background
tmp = arr[:,:360]
plt.imshow(image.array_to_img(tmp,scale=False))
plt.show()

yesList = np.reshape(tmp,(-1,3))

tmp.shape
yesList.shape


# Below we build seen dictionary of the unique pixel colors in yesList. 
# For all keys in the dictionary, the value is 1.
seen = {}
for c in yesList:
  col = str(c)
  if col not in seen: seen[col] = 1
len(seen)

# foreground
tmp = arr[30:,547:620]
plt.imshow(image.array_to_img(tmp,scale=False))
plt.show()

noList = np.reshape(tmp,(-1,3))


arr.shape

# Build a list of pixels for both positive and negative examples.
alldat = np.concatenate((yesList,noList))

# labels
labs = np.concatenate((np.ones(len(yesList)), np.zeros(len(noList))))


# Add an additional column to the data corresponding to
#  the offset parameter.
alldat = np.concatenate((alldat,np.ones((len(alldat),1))),1)

# Compute the loss of the rule specified by weights w with respect
#  to the data alldat labeled with labs
def loss(w, alldat, labs):
  # Compute a weighted sum for each instance
  h = np.matmul(alldat,w)
  # transform the sum using the sigmoid function
  y = 1/(1 + np.exp(-h))
  # take the difference between the labels and the output of the
  #  sigmoid, squared, then sum up over all instances to get the
  #  total loss.
  loss = np.sum((labs - y)**2)
  return(loss)


# repeat 10 times
for i in range(10):
  # pick a random vector of weights, with values between -1 and 1
  w = np.random.random(4)*2-1
  # report the loss
  print(w, loss(w, alldat, labs))

def fit(w,alldat,labs):
  # alpha represents how big of a step we’ll
  #  be taking in the direction of the derivative.
  #  It’s called the learning rate.
  alpha = 0.1

  # We'll stop searching when we're at a (near) local min
  done = False
  while not done:
    # Every 100 iterations or so, let’s
    #  take a peek at the weights, the learning
    #  rate, and the current loss
    if np.random.random() < 0.01: print(w, alpha, loss(w,alldat,labs))
    # The next few lines compute the gradient
    #  of the loss function. The details aren’t
    #  important right now.
    # delta_w is the change in the weights
    #  suggested by the gradient
    h = np.matmul(alldat,w)
    y = 1/(1 + np.exp(-h))
#    delta_w = np.add.reduce(np.reshape((labs-y) * np.exp(-h)/(1 + np.exp(-h))**2,(len(y),1)) * alldat)
    delta_w = np.add.reduce(np.reshape((labs-y) * np.exp(-h)*y**2,(len(y),1)) * alldat)
    # if we take a step of size alpha and update
    #  the weights, we’ll get new weights neww.
    current_loss = loss(w,alldat,labs)
    alpha *= 2
    neww = w + alpha* delta_w
    while loss(neww,alldat,labs) >= current_loss and not done:
      alpha /= 2
      if alpha*max(abs(delta_w)) < 0.0001:
        done = True
        print(alpha,delta_w)
      else: neww = w + alpha* delta_w
    if not done: w = neww
  return(w)

# w = fit([-2.0, 0, 0.093, -0.713],alldat,labs)

w = [ 0.786,  0.175, -0.558, -0.437]
w = fit(w,alldat,labs)

print(loss([-0.138, -1.62, -1.00, -1.00], alldat, labs))
print(loss(w, alldat, labs))
print(w)


# Turn the pixels in the image into a list
flat = np.reshape(arr,(-1,3))
# Stick a "1" at the end of each color
flat = np.concatenate((flat,np.ones((len(flat),1))),1)
# Multiply by the pixels by the weight matrix,
#  and set a threshold of 0.
out = np.matmul(flat, w) > 0.0
# Reshape the output as a 2 dimensional list instead of 1 dimensional
out = np.reshape(out,(-1,1))
# Now, concatenate this list it itself three times to make something
#  like a color. Reshape the resulting list into the shape of the original
#  image.
newarr = np.reshape(np.concatenate((out, out, out),1),arr.shape)
# Display the image
plt.imshow(image.array_to_img(newarr))
plt.show()

img = keras.utils.load_img("forest.jpg")

plt.imshow(img)
plt.show()

bkg = image.img_to_array(img)

def composite(mask, foreground, background):
  ishift = 157
  print(mask.shape)
  for i in range(min(background.shape[0],foreground.shape[0]+ishift)):
    for j in range(min(background.shape[1], foreground.shape[1])):
      fgi = i - ishift
#      if not mask[i][j][0]: background[i][j] = foreground[i][j]
      if fgi >= 0 and not mask[fgi][j][0]: background[i][j] = foreground[fgi][j]
  plt.imshow(img)
  plt.show()


composite(newarr,arr,bkg)