# USIM Prototype

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
VEC <- 100
ggplot(data = EV, aes(x, y)) + geom_tile(aes(fill = EV[,VEC+2])) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Split in train and test sets
set.seed(10)
list <- sort(sample(1:4999, 1667))

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

# Subset train and test samples
coords.train <- coords[-list,]
coords.test <- coords[list,]
ggplot(data = EV[-list,], aes(x = x, y = y)) + geom_tile(aes(fill = MEM100)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Organize training data by nearest neighbors
# Find the k = 8 nearest neighbors
g8nn.train <- knn2nb(knearneigh(as.matrix(coords.train), k = 8), row.names = ID[-list])
# These arrays will contain 3,332 maps for training, each with 3,333 observations
VAR8nn.train <- array(data = NA, dim = c(3332*3333,8))
VAR.train <- array(data = NA, dim = c(3332*3333))
MC.train <- array(data = NA, dim = c(3332*3333))
# Select the maps for training
VARS <- EV[,-list]
count <- 0
for(j in 1:3332){
  # Select map and observations for creating the training data set
  evec <- VARS[,j+2]
  evec <- evec[-list]
  for(i in 1:3333){
    count <- count + 1
    VAR.train[count] <- evec[i]
    MC.train[count] <- mc.train[j]
    VAR8nn.train[count,] <- VARS[attributes(g8nn.train)$region.id[g8nn.train[[i]]], j+2]
  }
  print(j)
}

ggplot(data = EV[attributes(g8nn.train)$region.id[g8nn.train[[200]]],], aes(x = x, y = y)) + geom_tile(aes(fill = MEM1)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Build the network

#Main input layer: 8 nearest neighbors of training data
main_input <- layer_input(shape = c(8), name = "main_input")

# First dense network
dense1_output <- main_input %>% 
  layer_dense(units = 8, activation = "softmax", input_shape = c(8)) %>% 
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dense(units = 20, activation = 'softmax') %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dense(units = 15, activation = 'softmax')
  
# Auxiliary output
auxiliary_output <- dense1_output %>%
  layer_dense(units = 1, activation = "relu", name = "aux_output")

# Auxiliary input layer: Moran's coefficient of input map
auxiliary_input <- layer_input(shape = c(1), name = "aux_input")

# Main output: concatenate the auxiliary input to the output of dense1 output
dense2_output <- layer_concatenate(c(dense1_output, auxiliary_input)) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1, activation = "relu", name = "main_output")

usim <- keras_model(
  inputs = c(main_input,auxiliary_input),
  outputs = c(dense2_output, auxiliary_output)
)

summary(usim)

# Compile model
usim %>% compile(
  optimizer = "adagrad",
  loss = "mae",
  loss_weights = c(1.0, 0.2),
  metrics = c("mae", "acc")
)

# Train model
usim %>% fit(
  x = list(VAR8nn.train, MC.train),
  y = list(VAR.train, VAR.train),
  epochs = 20,
  batch_size = 32
)

