#Load libraries
library(spdep)
library(adespatial)
library(dplyr)
library(ggplot2)
library(keras)

rm(list = ls())

# Create a grid
coords <- expand.grid(x=seq(1,100,by=2), y=seq(1,200,by=2))
ID <- seq(1, 5000, by=1)

# find the k = 8 nearest neighbors
g8nn <- knn2nb(knearneigh(as.matrix(coords), k = 8), row.names = ID)
# create a listw object (an adjacency matrix in list form)
g8listw <- nb2listw(g8nn)

# Calculate eigenvectors
EV <- mem(g8listw)
# Normalize the eigenvectors
maxs <- apply(EV, 2, max) 
mins <- apply(EV, 2, min)
EV <- as.data.frame(scale(EV, center = mins, scale = maxs - mins))
# Append the coordinates
EV <- cbind(coords, EV)
# Plot eigenvector  
VEC <- 15
ggplot(data = EV, aes(x, y)) + geom_tile(aes(fill = EV[,VEC+2])) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Artificial Neural Network for predicting Moran's coefficient given a map

# Flattened data needs to be converted to tensors
EV.t <- t(data.matrix(EV[,3:5001]))
# Redimension the training and test images
dim(EV.t) <- c(4999,50,100,1)
EV.t <- aperm(EV.t, c(1,3,2,4))

# Split in train and test sets
set.seed(10)
list <- sort(sample(1:4999, 1667))
EV.t.train <- EV.t[-list,,,]
dim(EV.t.train) <- c(3332,100,50,1)
EV.t.test <- EV.t[list,,,]
dim(EV.t.test) <- c(1667,100,50,1)

# Calculate Moran's coefficient for eigenvectors
mc <- numeric()
for(i in 3:5001){
  junk <- moran(EV[,i], g8listw, length(g8nn), Szero(g8listw))
  mc[i-2] <- as.numeric(junk[1])
}
# Plot Moran's coefficients
plot(mc)

# Split in train and test sets
mc.train <- mc[-list]
mc.test <- mc[list]

# Initialize the model
model <- keras_model_sequential()

# Define the model
model %>%
  layer_conv_2d(filters = 20, kernel_size = c(2,2), activation = 'relu',
                input_shape = c(100, 50,1)) %>%
  layer_conv_2d(filters = 20, kernel_size = c(2,2), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 20, kernel_size = c(2,2), activation = 'relu') %>%
  layer_conv_2d(filters = 20, kernel_size = c(2,2), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>%
  #layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = 'tanh')

# Compile the model. Note the use of an alias for the loss function
model %>%
  compile(
    loss = 'mae',
    optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = TRUE)
  )

# Train the model
history <- model %>% fit(EV.t.train, mc.train, batch_size = 10, epochs = 20)

# Evauate the model
score <- model %>% evaluate(EV.t.test, mc.test, batch_size = 10)
score

# Predict
predicted_output <- model %>% predict(EV.t.test, batch_size = 10)
# Plot predictions against observations
comp <- as.data.frame(cbind(mc.test,predicted_output))
comp <- rename(comp, predicted = V2)
ggplot(data = comp, aes(x = mc.test, y = predicted)) + geom_point() + geom_abline(intercept = 0, slope = 1)

## Artificial Neural Network for spatial prediction

# Subset train and test samples
coords.train <- coords[-list,]
coords.test <- coords[list,]
ggplot(data = EV[-list,], aes(x = x, y = y)) + geom_tile(aes(fill = MEM1)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Organize training data by nearest neighbors
# Find the k = 8 nearest neighbors for train observations
g8nn.train <- knn2nb(knearneigh(as.matrix(coords.train), k = 8), row.names = ID[-list])
# Assemble list of nearest neighbors and observations for train observations
VAR <- EV$MEM1
VAR.train <- VAR[-list]
VAR8nn.train <- array(data = NA, dim = c(3333,8))
for(i in 1:3333){
  #NOTE: the list of neighbors is built using the region.idsffdf
  VAR8nn.train[i,] <- VAR[attributes(g8nn.train)$region.id[g8nn.train[[i]]]]
}
ggplot(data = EV[attributes(g8nn.train)$region.id[g8nn.train[[3000]]],], aes(x = x, y = y)) + geom_tile(aes(fill = MEM1)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Create model
model_sp <- keras_model_sequential()

# Define and compile the model
model_sp %>% 
  layer_dense(units = 8, activation = 'softmax', input_shape = c(8)) %>% 
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dense(units = 20, activation = 'softmax') %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dense(units = 15, activation = 'softmax') %>%
  layer_dense(units = 1, activation = 'relu')

model_sp %>% 
  compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = "acc")

# save_model_hdf5(model_sp, "model_sp_8s_20r_20s_15r_15s_1r", overwrite = TRUE,
#                 include_optimizer = TRUE)

# Train the model
history <- model_sp %>% fit(VAR8nn.train, VAR.train, batch_size = 10, epochs = 20)

# Organize test data by nearest neighbors
# Find the k = 8 nearest neighbors for test observations
id.junk <- ID[-list]
coords.junk <- coords.train # Coordinates of train observations
g8nn.test <- list() # Initialize list of nearest neighbors
for(i in 1:1667){
  id.junk[3334] <- ID[list[i]] # Retrieve the region.id for the ith item in the test sample
  coords.junk[3334,] <- coords[list[i],] # Append the coordinates of the ith item of test sample
  junk <- knn2nb(knearneigh(as.matrix(coords.junk), k = 8), row.names = id.junk) # Find nearest neighbors
  #NOTE: This list of nearest neighbors is built using the region.ids
  g8nn.test[[i]] <- attributes(junk)$region.id[junk[[3334]]] # Retrieve nearest neighbors of ith item in test sample
  print(i)
}
# Assemble list of nearest neighbors and observations for test observations
VAR <- EV$MEM100
VAR.test <- VAR[list]
VAR8nn.test <- array(data = NA, dim = c(1667,8))
for(i in 1:1667){
  VAR8nn.test[i,] <- VAR[g8nn.test[[i]]] 
}
ggplot(data = EV[g8nn.test[[3]],], aes(x = x, y = y)) + geom_tile(aes(fill = MEM100)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Evauate the model
score <- model_sp %>% evaluate(VAR8nn.test, VAR.test, batch_size = 10)
score

# Predict
pred_sp <- model_sp %>% predict(VAR8nn.test, batch_size = 10)
#Plot predictions against observations
comp_sp <- as.data.frame(cbind(VAR.test,pred_sp))
comp_sp <- rename(comp_sp, predicted = V2)
ggplot(data = comp_sp, aes(x = VAR.test, y = predicted)) + geom_point() + geom_abline(intercept = 0, slope = 1)
cor(VAR.test,pred_sp)


# Systematically test the fit of the predictions using all maps
corr_sp <- numeric()
for(i in 1:4999){
  VAR <- EV[,i+2]
  VAR.test <- VAR[list]
  VAR8nn.test <- array(data = NA, dim = c(1667,8))
  for(j in 1:1667){
    VAR8nn.test[j,] <- VAR[g8nn.test[[j]]] 
  }
  # Predict
  pred_sp <- model_sp %>% predict(VAR8nn.test, batch_size = 10)
  #Plot predictions against observations
  comp_sp <- as.data.frame(cbind(VAR.test,pred_sp))
  comp_sp <- rename(comp_sp, predicted = V2)
  corr_sp[i] <- as.numeric(cor(VAR.test,pred_sp))
  print(i)
}
  


